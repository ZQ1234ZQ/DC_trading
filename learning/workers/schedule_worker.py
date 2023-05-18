#显示进度条，收集Final_return跟experience
#Display progress bar, collect Final_return and experience

import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum
from multiprocessing.managers import SyncManager

from rich.console import Group
from rich.live import Live
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from setproctitle import setproctitle

from learning.utils import console


class Action(Enum):
    REGISTER = 0
    UPDATE = 1
    FINISH = 2


@dataclass
class Content:
    episode_id: int
    num_steps: int = None
    result: any = None


@dataclass
class Message:
    connection_id: int
    action: Action
    content: Content = None


class _ScheduleClient:
    def __init__(self, connection_id, episode_id, input_queue) -> None:
        self._connection_id = connection_id
        self._episode_id = episode_id
        self._input_queue = input_queue

    @property
    def episode_id(self):
        return self._episode_id

    def _register(self):
        self._input_queue.put(
            Message(self._connection_id, Action.REGISTER, Content(self._episode_id))
        )

    def update(self, num_steps):
        self._input_queue.put(
            Message(
                self._connection_id,
                Action.UPDATE,
                Content(self._episode_id, num_steps=num_steps),
            )
        )

    def finish(self, result):
        self._input_queue.put(
            Message(
                self._connection_id,
                Action.FINISH,
                Content(self._episode_id, result=result),
            )
        )


class ScheduleWorker(mp.Process):
    def __init__(
        self, desc, num_episodes, max_num_steps, readys=[], num_parallel=None
    ) -> None:
        super().__init__()
        self._desc = desc
        self._num_episodes = num_episodes
        self._max_num_steps = max_num_steps
        self._readys = readys

        if num_parallel is None:
            num_parallel = mp.cpu_count()
        self._connection_ids = mp.SimpleQueue()
        for connection_id in range(num_parallel):
            self._connection_ids.put(connection_id)

        self._input_queue = mp.SimpleQueue()

        sync_manager = SyncManager()
        sync_manager.start(lambda: setproctitle("sync-manager"))
        self._results = sync_manager.list()

    def __enter__(self):
        self.start()
        return self

    def register(self, episode_id):
        connection_id = self._connection_ids.get()
        client = _ScheduleClient(connection_id, episode_id, self._input_queue)
        client._register()
        return client

    def run(self) -> None:
        setproctitle("schedule-worker")

        for ready in self._readys:
            ready()

        episodes_progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
        )
        steps_progress = Progress(
            TextColumn("    "),
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
        )

        with Live(Group(episodes_progress, steps_progress), console=console):
            episodes_task = episodes_progress.add_task(
                self._desc, total=self._num_episodes
            )
            steps_task_id_dict = {}
            num_steps_dict = {}
            while True:
                message = self._input_queue.get()
                if message.action == Action.REGISTER:
                    task_id = steps_progress.add_task(
                        f"Episode {message.content.episode_id}",
                        total=self._max_num_steps,
                    )
                    steps_task_id_dict[message.content.episode_id] = task_id
                elif message.action == Action.UPDATE:
                    num_steps_dict[
                        message.content.episode_id
                    ] = message.content.num_steps
                    steps_progress.update(
                        steps_task_id_dict[message.content.episode_id],
                        completed=num_steps_dict[message.content.episode_id],
                    )
                elif message.action == Action.FINISH:
                    steps_progress.update(
                        steps_task_id_dict[message.content.episode_id],
                        total=num_steps_dict[message.content.episode_id],
                        visible=False,
                    )
                    self._results.append(message.content.result)
                    episodes_progress.update(episodes_task, advance=1)
                    self._connection_ids.put(message.connection_id)

                    if episodes_progress.finished:
                        break

    def __exit__(self, *args):
        self.join()

    @property
    def results(self):
        self.join()
        return self._results

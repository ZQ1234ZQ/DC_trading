#调用神经网络，包含两个对象，一个cs结构的进程间通信的程序。程序启动时会启动24（CPU核心个）个_PredictClient跟1个PredictWorker。execute_episode_worder掉用_PredictClient
#并把请求汇总给PredictWorker然后由PredictWorker统一调用GPU运算给出结果，只有PredictWorker能打开神经网络调用
#Call the neural network, including two objects, a cs-structured inter-process communication program. 
#When the program starts, 24 (number of CPU cores) _PredictClient and 1 PredictWorker will be started. execute_episode_worder use_PredictClient
#And summarize the requests to PredictWorker, and then PredictWorker will uniformly call the GPU operation to give the results. Only PredictWorker can open the neural network call
import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

import numpy as np
from setproctitle import setproctitle

from learning.nnet import NeuralNet


class Action(Enum):
    REGISTER = 0
    PREDICT = 1
    SKIP = 2
    CLOSE = 3


@dataclass
class Content:
    model_name: str
    observation: np.ndarray


@dataclass
class Message:
    connection_id: int
    action: Action
    content: Content = None


class _PredictClient:
    def __init__(self, connection_id, model_names, input_queue, output_queue) -> None:
        self._connection_id = connection_id
        self._model_names = model_names
        self._input_queue = input_queue
        self._output_queue = output_queue

    @property
    def model_names(self):
        return self._model_names

    def _register(self):
        self._input_queue.put(Message(self._connection_id, Action.REGISTER))

    def predict(self, model_name):
        def _predict(observation):
            self._input_queue.put(
                Message(
                    self._connection_id,
                    Action.PREDICT,
                    Content(model_name, observation),
                )
            )
            return self._output_queue.get()

        return _predict

    def skip(self):
        self._input_queue.put(Message(self._connection_id, Action.SKIP))
        return self._output_queue.get()

    def close(self):
        self._input_queue.put(Message(self._connection_id, Action.CLOSE))


class PredictWorker(mp.Process):
    def __init__(self, model_names, batch_size=None) -> None:
        super().__init__()
        self._model_names = model_names

        if batch_size is None:
            batch_size = mp.cpu_count()
        self._connection_ids = mp.SimpleQueue()
        for connection_id in range(batch_size):
            self._connection_ids.put(connection_id)

        self._ready = mp.Condition()

        self._input_queue = mp.SimpleQueue()
        self._output_queues = [mp.SimpleQueue() for _ in range(batch_size)]

    def __enter__(self):
        self.start()
        return self

    def ready(self):
        with self._ready:
            self._ready.wait()

    def register(self):
        connection_id = self._connection_ids.get()
        client = _PredictClient(
            connection_id,
            self._model_names,
            self._input_queue,
            self._output_queues[connection_id],
        )
        client._register()
        return client

    def run(self) -> None:
        setproctitle("predict-worker")

        nnet = {
            model_name: NeuralNet.load(model_name) for model_name in self._model_names
        }
        with self._ready:
            self._ready.notify_all()

        num_alive = 0
        inputs = defaultdict(list)
        num_inputs = 0
        skips = []
        while True:
            message = self._input_queue.get()
            if message.action == Action.REGISTER:
                num_alive += 1
            elif message.action == Action.PREDICT:
                inputs[message.content.model_name].append(
                    (message.connection_id, message.content.observation)
                )
                num_inputs += 1
            elif message.action == Action.SKIP:
                skips.append(message.connection_id)
                num_inputs += 1
            elif message.action == Action.CLOSE:
                num_alive -= 1
                self._connection_ids.put(message.connection_id)

            if num_inputs == num_alive:
                for model_name in self._model_names:
                    if len(inputs[model_name]) > 0:
                        connection_ids, observations = zip(*inputs[model_name])
                        qs = nnet[model_name].predict(observations)
                        for i, connection_id in enumerate(connection_ids):
                            q = qs[i]
                            self._output_queues[connection_id].put((q))
                        inputs[model_name] = []

                for connection_id in skips:
                    self._output_queues[connection_id].put(None)
                skips = []

                num_inputs = 0

    def __exit__(self, *args):
        self.terminate()
        self.join()

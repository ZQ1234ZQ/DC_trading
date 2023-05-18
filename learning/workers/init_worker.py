#install network


import multiprocessing as mp

from setproctitle import setproctitle

from learning.nnet import NeuralNet


class InitWorker(mp.Process):
    def __init__(self, model_name, env_id) -> None:
        super().__init__()
        self._model_name = model_name
        self._env_id = env_id

    def run(self) -> None:
        setproctitle("init-worker")

        if not NeuralNet.exists(self._model_name):
            nnet = NeuralNet.create(self._model_name, self._env_id)
            nnet.save(self._model_name)

import multiprocessing as mp

from setproctitle import setproctitle

from learning.nnet import NeuralNet


class UpdateWorker(mp.Process):
    def __init__(self, model_name, new_model_name) -> None:
        super().__init__()
        self._model_name = model_name
        self._new_model_name = new_model_name

    def run(self) -> None:
        setproctitle("update-worker")

        new_nnet = NeuralNet.load(self._new_model_name)
        new_nnet.save(self._model_name)

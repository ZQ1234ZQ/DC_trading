#Training model show the reward, loss function, 

import multiprocessing as mp

import tensorflow as tf
from keras.losses import MSE
from keras.optimizers import Adam
from setproctitle import setproctitle

from learning.experience import ExperienceManager
from learning.nnet import NeuralNet


class TrainWorker(mp.Process):
    def __init__(self, model_name, new_model_name, tau=1e-3) -> None:
        super().__init__()
        self._model_name = model_name
        self._new_model_name = new_model_name
        self._tau = 1e-3

    def compute_loss(self, data, nnet, target_nnet, gamma=0.995):
        (
            states,
            actions,
            rewards,
            next_states,
            done_vals,
        ) = data
        max_qsa = tf.reduce_max(target_nnet._model(next_states), axis=-1)
        y_targets = rewards #+ gamma * max_qsa * (1 - done_vals)
        q_values = nnet._model(states)
        q_values = tf.gather_nd(
            q_values,
            tf.stack([tf.range(q_values.shape[0]), tf.cast(actions, tf.int32)], axis=1),
        )
        loss = MSE(y_targets, q_values)   #最小二乘法 mean square error
        return loss

    @tf.function#计算梯度，加速运算，可有可无
    def learn(self, data, nnet, target_nnet, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(data, nnet, target_nnet)

        gradients = tape.gradient(loss, nnet._model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, nnet._model.trainable_variables))

    def run(self) -> None:
        setproctitle("train-worker")

        nnet = NeuralNet.load(self._model_name)
        target_nnet = NeuralNet.load(self._model_name)
        experience_manager = ExperienceManager(self._model_name)
        experiences = experience_manager.load()
        data = experience_manager.unpack(experiences)
        self.learn(data, nnet, target_nnet, nnet.optimizer)
        for target_weights, q_net_weights in zip(
            target_nnet._model.weights, nnet._model.weights
        ):
            target_weights.assign(
                self._tau * q_net_weights + (1.0 - self._tau) * target_weights
            )
        target_nnet.save(self._model_name)

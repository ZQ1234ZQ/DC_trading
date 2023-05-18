#setting network


import os
from functools import reduce

import numpy as np
from keras import Model
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
)
from keras.optimizers import Adam
from keras.regularizers import l2
from tensorflow import keras

from learning.envs import env_cls
from learning.utils import console


class NeuralNet:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    def __init__(self, model: Model) -> None:
        self._model = model

    @staticmethod
    def _Conv2D(filters, kernel_size):
        return Conv2D(
            filters,
            kernel_size,
            padding="same",
            use_bias=False,
            kernel_regularizer=l2(1e-4),
        )

    @staticmethod
    def _Dense(units, activation, name=None):
        return Dense(units, activation, kernel_regularizer=l2(1e-4), name=name)

    @classmethod
    def create(cls, model_name, env_id, num_residual_layers=7):
        with console.status(f"Creating model {model_name}..."):
            x_input = Input(env_cls(env_id).observation_shape, name="observation_input")
            input_layers = [
                x_input,
                cls._Conv2D(filters=256, kernel_size=5),
                BatchNormalization(axis=1),
                Activation("relu"),
            ]
            x = reduce(lambda x, y: y(x), input_layers)

            for _ in range(num_residual_layers):
                residual_layers = [
                    x,
                    cls._Conv2D(filters=256, kernel_size=3),
                    BatchNormalization(axis=1),
                    Activation("relu"),
                    cls._Conv2D(filters=256, kernel_size=3),
                    BatchNormalization(axis=1),
                ]
                x_residual = reduce(lambda x, y: y(x), residual_layers)

                x = Add()([x, x_residual])
                x = Activation("relu")(x)

            policy_output_layers = [
                x,
                cls._Conv2D(filters=2, kernel_size=1),
                BatchNormalization(axis=1),
                Activation("relu"),
                Flatten(),
                Dense(env_cls(env_id).num_actions),
            ]
            policy_output = reduce(lambda x, y: y(x), policy_output_layers)

            model = Model(x_input, [policy_output], name=model_name)

            model.compile(
                Adam(),
                ["categorical_crossentropy", "mean_squared_error"],
                loss_weights=[1.25, 1.0],
            )

        console.log(f"Created model {model_name}.")

        return cls(model)

    def predict(self, observation):
        x = np.array(observation)

        input_shape = self._model.layers[0].input_shape.pop()
        need_reshape = x.ndim < len(input_shape)

        if need_reshape:
            x = np.expand_dims(x, axis=0)

        p = self._model.predict(x, len(x), False)
        if need_reshape:
            p = p.squeeze(axis=0)

        return p

    @property
    def optimizer(self, alpha=1e-3):
        return Adam(learning_rate=alpha)

    @staticmethod
    def _model_path(model_name):
        return os.path.join(os.path.dirname(__file__), "models", model_name + ".h5")

    def save(self, model_name):
        path = self._model_path(model_name)
        with console.status(f"Saving model into {path}..."):
            self._model.save(path)
        console.log(f"Saved model into {path}.")

    @classmethod
    def exists(cls, model_name):
        path = cls._model_path(model_name)
        is_exists = os.path.exists(path)
        if is_exists:
            console.log(f"Model was found at {path}.")
        else:
            console.log("Model was not found.")
        return is_exists

    @classmethod
    def load(cls, model_name):
        path = cls._model_path(model_name)
        with console.status(f"Loading model from {path}..."):
            model = keras.models.load_model(path)
        console.log(f"Loaded model from {path}.")
        return cls(model)

    @classmethod
    def destroy(cls, model_name):
        path = cls._model_path(model_name)
        with console.status(f"Deleting model from {path}..."):
            os.remove(path)
        console.log(f"Deleted model from {path}.")

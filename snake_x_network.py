import numpy as np
import tensorflow.keras as K
import tensorflow as tf




class Network:
    def __init__(self, arg):
        self.img_shape = arg['img_shape']
        self.action_num = arg['action_num']
        self.lr = arg['learning_late']

    def make_model(self, load):
        normal = K.initializers.glorot_normal()
        inputs = K.layers.Input(shape=self.img_shape+(1,))
        x = K.layers.Conv2D(64, kernel_size=8, strides=4, padding="same", kernel_initializer=normal, activation="relu")(inputs)
        x = K.layers.Conv2D(128, kernel_size=4, strides=2, padding="same", kernel_initializer=normal, activation="relu")(x)
        x = K.layers.Conv2D(128, kernel_size=3, strides=1, padding="same", kernel_initializer=normal, activation="relu")(x)
        x = K.layers.Flatten()(x)

        #Advantage
        A = K.layers.Dense(512, kernel_initializer=normal, activation="relu")(x)
        A = K.layers.Dense(self.action_num, kernel_initializer=normal)(A)
        #V
        V = K.layers.Dense(512, kernel_initializer=normal, activation="relu")(x)
        V = K.layers.Dense(1, kernel_initializer=normal)(V)

        y = K.layers.concatenate([V,A])
        outputs = K.layers.Lambda(lambda a: a[:,:1] + a[:,1:] - tf.stop_gradient(K.backend.mean(a[:,1:], keepdims=True,axis=1)), output_shape=(self.action_num,))(y)

        self.model = K.Model(inputs=inputs, outputs=outputs)
        #load model
        if load != "":
            self.model.load_weights("param.hdf5")
        self.model.compile(optimizer=K.optimizers.Adam(lr=self.lr, clipvalue=1.0), loss="mean_squared_error")

        self.target_model = K.Model(inputs=inputs, outputs=outputs)
        self.target_model.compile(optimizer=K.optimizers.Adam(lr=self.lr, clipvalue=1.0), loss="mean_squared_error")

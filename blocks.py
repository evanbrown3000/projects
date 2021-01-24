import keras
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, MaxPool2D, Convolution2DTranspose
from tensorflow.keras.layers import ZeroPadding2D, Reshape, ReLU, Add, Concatenate

class DepthMaxPool(keras.layers.Layer):
    def __init__(self, pool_size, strides=None, padding="VALID", **kwargs):
        super().__init__(**kwargs)
        if strides is None:
            strides = pool_size
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
    def call(self, inputs):
        return tf.nn.max_pool(inputs,
                              ksize=(1, 1, 1, self.pool_size),
                              strides=(1, 1, 1, self.pool_size),
                              padding=self.padding)
#TODO *** RUN INSIDE CUSTOM TRAIN FXN***
# with tf.device("/cpu:0"): # there is no GPU-kernel yet
#     Z = DepthMaxPool(DEPTH_POOL_SIZE)(Z)

class SubBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, activation='relu', rate = 0.2, **kwargs):
        super().__init__(**kwargs)
        
        self.activation = keras.activations.get(activation)
        
        self.layers = [
                keras.layers.Conv2D(filters, kernel_size, strides=strides, activation = None, padding= 'SAME'),
                keras.layers.BatchNormalization(),
                self.activation,
                keras.layers.Dropout(rate)
        ]

        

    def call(self, inputs):
        Z = inputs
        for layer in self.layers:
            Z = layer(Z)
        return Z

class SuperBlock(keras.layers.Layer):
    def __init__(self, n_sub_blocks, filters, kernel_size, strides,activation='relu', rate = 0.2, **kwargs):
        super().__init__(**kwargs)

        self.main_layers = [SubBlock(filters,kernel_size, strides=strides, activation=activation,rate=rate) for layer in range(n_sub_blocks) ]

        self.skip_layers = [
            keras.layers.Conv2D(filters,kernel_size,strides=strides,activation=None, padding= 'SAME',use_bias=False),
            keras.layers.BatchNormalization(),
        ]

        self.final_layers = [
            keras.layers.Dropout(rate)
        ]
        self.activation = keras.activations.get(activation)

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)

        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)

        Z = self.activation(Z + skip_Z)
        for layer in self.final_layers:
            Z = layer(Z)
        return Z

class ConvBlock(keras.layers.Layer):
    def __init__(self, kernel_size, strides, filters, activation='relu', rate = 0.2,padding='SAME', **kwargs):
        super().__init__(**kwargs)

        self.activation = keras.activations.get(activation)

        self.layers = [
                keras.layers.Conv2D(kernel_size=kernel_size, strides=strides, filters=filters, activation = None, padding= padding),
                keras.layers.BatchNormalization(),
                self.activation,
                keras.layers.Dropout(rate)
        ]


    def call(self, inputs):
        Z = inputs
        for layer in self.layers:
            Z = layer(Z)
        return Z

class DenseBlock(keras.layers.Layer):
    def __init__(self, neurons, activation='relu', rate = 0.2, **kwargs):
        super().__init__(**kwargs)

        self.activation = keras.activations.get(activation)

        self.layers = [
                keras.layers.Dense(neurons, activation = None),
                keras.layers.BatchNormalization(),
                self.activation,
                keras.layers.Dropout(rate)
        ]


    def call(self, inputs):
        Z = inputs
        for layer in self.layers:
            Z = layer(Z)
        return Z

class ConvBlock(keras.layers.Layer):
    def __init__(self, kernel_size, strides, filters, activation='relu', rate = 0.2,padding='SAME', **kwargs):
        super().__init__(**kwargs)

        self.activation = keras.activations.get(activation)

        self.layers = [
                keras.layers.Conv2D(kernel_size=kernel_size, strides=strides, filters=filters, activation = None, padding= padding),
                keras.layers.BatchNormalization(),
                self.activation,
                keras.layers.Dropout(rate)
        ]


    def call(self, inputs):
        Z = inputs
        for layer in self.layers:
            Z = layer(Z)
        return Z

class AvgPoolBlock(keras.layers.Layer):
    def __init__(self, pool_size, strides, padding='VALID',rate = 0.2, **kwargs):
        super().__init__(**kwargs)

        self.layers = [
                keras.layers.AveragePooling2D(pool_size=pool_size, strides=strides,padding= padding),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(rate)
        ]

    def call(self, inputs):
        Z = inputs
        for layer in self.layers:
            Z = layer(Z)
        return Z

def residual_block(tensor, feature_n,name=None):
    if name != None:
        dconv = DepthwiseConv2D(3,padding='same',name=name+"/dconv")(tensor) 
        conv  = Conv2D(feature_n,1,padding='valid',name=name+"/conv")(dconv)
    else:
        dconv = DepthwiseConv2D(3,padding='same')(tensor) 
        conv  = Conv2D(feature_n,1,padding='valid')(dconv)
    add   = Add()([conv, tensor])
    relu = ReLU()(add)
    return relu 
def residual_block_id(tensor, feature_n,name=None):
    if name != None:
        depconv_1  = DepthwiseConv2D(3,2,padding='same',name=name+"/dconv")(tensor)
        conv_2     = Conv2D(feature_n,1,name=name+"/conv")(depconv_1)
    else:
        depconv_1  = DepthwiseConv2D(3,2,padding='same')(tensor)
        conv_2     = Conv2D(feature_n,1)(depconv_1)


    maxpool_1  = MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same')(tensor)
    conv_zeros = Conv2D(feature_n/2,2,strides=2,use_bias=False,kernel_initializer=tf.zeros_initializer())(tensor)

    padding_1  = Concatenate(axis=-1)([maxpool_1,conv_zeros])#self.feature_padding(maxpool_1)

    add = Add()([padding_1,conv_2])
    relu = ReLU()(add)

    return relu

class MaxPoolBlock(keras.layers.Layer):
    def __init__(self, pool_size, strides, padding='VALID',rate = 0.2, **kwargs):
        super().__init__(**kwargs)

        self.layers = [
                keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides,padding= padding),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(rate)
        ]

    def call(self, inputs):
        Z = inputs
        for layer in self.layers:
            Z = layer(Z)
        return Z
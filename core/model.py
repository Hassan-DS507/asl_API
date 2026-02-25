import tensorflow as tf
from core.layers import *
from core.preprocess import CHANNELS, MAX_LEN, NUM_CLASSES

def get_model(max_len=MAX_LEN, dim=192):

    inp = tf.keras.Input((max_len, CHANNELS))
    x = inp

    ksize = 17

    x = tf.keras.layers.Dense(dim, use_bias=False, name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95, name='stem_bn')(x)

    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = TransformerBlock(dim, expand=2)(x)

    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = TransformerBlock(dim, expand=2)(x)

    x = tf.keras.layers.Dense(dim*2, name='top_conv')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(NUM_CLASSES, name='classifier')(x)

    return tf.keras.Model(inp, x)
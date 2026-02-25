import tensorflow as tf
from core.preprocess import Preprocess

class TFLiteModel(tf.Module):

    def __init__(self, models):
        super().__init__()
        self.prep = Preprocess()
        self.models = models

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32)
        ]
    )
    def __call__(self, inputs):

        x = self.prep(inputs)
        outputs = [m(x) for m in self.models]
        outputs = tf.reduce_mean(outputs, axis=0)

        return {"outputs": outputs}
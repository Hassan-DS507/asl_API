import tensorflow as tf
from inference.load_models import load_models
from inference.tflite_wrapper import TFLiteModel

paths = [
    "weights/seed42.h5",
    "weights/seed43.h5",
    "weights/seed44.h5"
]

models = load_models(paths)

wrapper = TFLiteModel(models)

converter = tf.lite.TFLiteConverter.from_keras_model(wrapper)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
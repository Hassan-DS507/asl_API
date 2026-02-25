import numpy as np
from inference.load_models import load_models

paths = [
    "weights/seed42_fold0_best.h5",
    "weights/seed43_fold0_best.h5",
    "weights/seed44_fold0_best.h5",
]

models = load_models(paths)

dummy = np.random.randn(1, 384, 708).astype("float32")

outputs = [m(dummy, training=False) for m in models]
outputs = np.mean(outputs, axis=0)

print("Prediction shape:", outputs.shape)
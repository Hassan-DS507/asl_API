import tensorflow as tf
import math

class CosineScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, total_steps, warmup_steps=0):
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.initial_lr * (step / self.warmup_steps)

        progress = (step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        return self.initial_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
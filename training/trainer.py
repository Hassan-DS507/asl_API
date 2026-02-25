import tensorflow as tf
from core.model import get_model
from configs.config import CFG

def build_model(steps_per_epoch):

    total_steps = steps_per_epoch * CFG.epochs

    from core.scheduler import CosineScheduler

    lr_schedule = CosineScheduler(
        initial_lr=CFG.lr,
        total_steps=total_steps,
        warmup_steps=1000
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model = get_model()

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    return model


def train(train_ds, valid_ds=None):

    steps_per_epoch = len(train_ds)

    model = build_model(steps_per_epoch)

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=CFG.epochs
    )

    model.save_weights("weights/final_model.h5")
import tensorflow as tf

class AWP:
    def __init__(self, model, optimizer, adv_lr=1e-4, adv_eps=1e-2):
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def attack(self):
        for var in self.model.trainable_variables:
            if var.dtype.is_floating:
                grad = self.optimizer.get_gradients(self.model.total_loss, [var])[0]
                if grad is None:
                    continue
                norm = tf.norm(grad)
                if norm != 0:
                    r_at = self.adv_lr * grad / (norm + 1e-6)
                    self.backup[var.name] = var.value()
                    var.assign_add(r_at)

    def restore(self):
        for var in self.model.trainable_variables:
            if var.name in self.backup:
                var.assign(self.backup[var.name])
        self.backup = {}
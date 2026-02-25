from core.model import get_model

def load_models(paths):

    models = []

    for p in paths:
        m = get_model()
        m.load_weights(p)
        models.append(m)

    return models
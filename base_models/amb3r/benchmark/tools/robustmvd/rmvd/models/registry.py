# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/models/registry.py

"""Model Registry.

Based on the model registry from the timm package ( https://github.com/rwightman/pytorch-image-models ).
"""


_model_entrypoints = {}
_trainable_model_entrypoints = {}


def register_model(arg=None, trainable=True):
    def _register_model(model_entrypoint):
        """Register a model."""
        model_name = model_entrypoint.__name__
        _model_entrypoints[model_name] = model_entrypoint
        if trainable:
            _trainable_model_entrypoints[model_name] = model_entrypoint
        return model_entrypoint
    
    if callable(arg): 
        return _register_model(arg)
    else:
        return _register_model


def list_models(trainable_only=False):
    """List all available models."""
    models = _model_entrypoints.keys() if not trainable_only else _trainable_model_entrypoints.keys()
    models = list(sorted(models))
    return models


def has_model(name, trainable_only=False):
    """Check if model is registered."""
    has_model = name in _model_entrypoints if not trainable_only else name in _trainable_model_entrypoints
    return has_model


def get_model(name):
    """Get model entrypoint by name."""
    assert has_model(name), f'The requested model "{name}" does not exist. Available models are: {" ".join(list_models())}'
    return _model_entrypoints[name]

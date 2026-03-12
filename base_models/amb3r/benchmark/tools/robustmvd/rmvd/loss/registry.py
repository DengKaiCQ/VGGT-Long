# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/loss/registry.py

"""Loss registry.

Based on the model registry from the timm package ( https://github.com/rwightman/pytorch-image-models ).
"""

_loss_entrypoints = {}


def register_loss(loss_entrypoint):
    """Register a loss."""
    loss_name = loss_entrypoint.__name__
    _loss_entrypoints[loss_name] = loss_entrypoint
    return loss_entrypoint


def list_losses():
    """List all available losses."""
    losses = _loss_entrypoints.keys()
    losses = list(sorted(losses))
    return losses


def has_loss(name):
    """Check if loss is registered."""
    return name in _loss_entrypoints


def get_loss(name):
    """Get loss entrypoint by name."""
    assert has_loss(name), f'The requested loss "{name}" does not exist. Available losses are: {" ".join(list_losses())}'
    return _loss_entrypoints[name]

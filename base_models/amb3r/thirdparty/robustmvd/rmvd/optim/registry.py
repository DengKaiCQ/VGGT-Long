# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/optim/registry.py

"""Optimizer registry.

Based on the model registry from the timm package ( https://github.com/rwightman/pytorch-image-models ).
"""


_optimizer_entrypoints = {}
_scheduler_entrypoints = {}


def register_optimizer(optimizer_entrypoint):
    """Register an optimizer."""
    optimizer_name = optimizer_entrypoint.__name__
    _optimizer_entrypoints[optimizer_name] = optimizer_entrypoint
    return optimizer_entrypoint


def list_optimizers():
    """List all available optimizers."""
    optimizers = _optimizer_entrypoints.keys()
    optimizers = list(sorted(optimizers))
    return optimizers


def has_optimizer(name):
    """Check if optimizer is registered."""
    return name in _optimizer_entrypoints


def get_optimizer(name):
    """Get optimizer entrypoint by name."""
    assert has_optimizer(name), f'The requested optimizer "{name}" does not exist. Available optimizers are: {" ".join(list_optimizers())}'
    return _optimizer_entrypoints[name]


def register_scheduler(scheduler_entrypoint):
    """Register a scheduler."""
    scheduler_name = scheduler_entrypoint.__name__
    _scheduler_entrypoints[scheduler_name] = scheduler_entrypoint
    return scheduler_entrypoint


def list_schedulers():
    """List all available schedulers."""
    schedulers = _scheduler_entrypoints.keys()
    schedulers = list(sorted(schedulers))
    return schedulers


def has_scheduler(name):
    """Check if scheduler is registered."""
    return name in _scheduler_entrypoints


def get_scheduler(name):
    """Get scheduler entrypoint by name."""
    assert has_scheduler(name), f'The requested scheduler "{name}" does not exist. Available schedulers are: {" ".join(list_schedulers())}'
    return _scheduler_entrypoints[name]

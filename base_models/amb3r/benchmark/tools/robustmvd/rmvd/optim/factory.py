# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/optim/factory.py

from .registry import get_optimizer, get_scheduler


def create_optimizer(name, model, **kwargs):
    """Creates an optimizer.

    Args:
        name (str): The name of the optimizer to create.
        model (torch.nn.Module): The model to optimize.

    Keyword Args:
        **kwargs: Additional arguments to pass to the optimizer.
    """
    optimizer_entrypoint = get_optimizer(name=name)
    optimizer = optimizer_entrypoint(model=model, **kwargs)
    optimizer.name = name
    return optimizer


def create_scheduler(name, optimizer, **kwargs):
    """Creates a scheduler.

    Args:
        name (str): The name of the scheduler to create.
        optimizer (torch.optim.Optimizer): The optimizer to schedule.

    Keyword Args:
        **kwargs: Additional arguments to pass to the scheduler.
    """
    scheduler_entrypoint = get_scheduler(name=name)
    scheduler = scheduler_entrypoint(optimizer=optimizer, **kwargs)
    scheduler.name = name
    return scheduler

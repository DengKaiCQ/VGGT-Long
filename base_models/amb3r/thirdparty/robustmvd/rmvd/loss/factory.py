# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/loss/factory.py

from .registry import get_loss


def create_loss(name, **kwargs):
    """Creates a loss.

    Args:
        name (str): The name of the loss to create.

    Keyword Args:
        **kwargs: Additional arguments to pass to the loss.
    """
    loss_entrypoint = get_loss(name=name)
    loss = loss_entrypoint(**kwargs)
    return loss

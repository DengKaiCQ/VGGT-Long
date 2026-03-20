# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/models/factory.py

import torch
import torch.nn as nn

from .registry import get_model
from .helpers import add_run_function


def create_model(name, pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    """Creates a model.

    Args:
        name (str): The name of the model to create.
        pretrained (bool): Whether to load the default pretrained weights for the model.
        weights (str): Path to custom weights to be loaded. Overrides `pretrained`.
        train (bool): Whether to put the model in train mode.
        num_gpus (int): Number of GPUs to be used from the model.

    Keyword Args:
        **kwargs: Additional arguments to pass to the model.
    """
    model_entrypoint = get_model(name=name)
    model = model_entrypoint(pretrained=pretrained, weights=weights, train=train, num_gpus=num_gpus, **kwargs)
    add_run_function(model)
    model.name = name
    return model


def prepare_custom_model(model, train=False, num_gpus=1):
    """Prepares a custom model for usage within the rmvd framework.

    The custom model must implement the input_adapter, forward/__call__, output_adapter functions.

    Args:
        model (nn.Module): The model to prepare.
        train (bool): Whether to put the model in train mode.
        num_gpus (int): Number of GPUs to be used from the model.
    """

    if train:
        model.train()
    else:
        model.eval()

    assert not isinstance(model, nn.DataParallel), 'Model must not be wrapped in nn.DataParallel before ' \
                                                   'prepare_custom_model is called.'

    if num_gpus == 1:
        model = model.cuda()
    elif num_gpus > 1:
        parallel_model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus))).cuda()
        parallel_model.input_adapter = model.input_adapter
        parallel_model.output_adapter = model.output_adapter
        model = parallel_model
    # elif num_gpus < 1: use CPU, nothing to be done

    add_run_function(model)
    return model

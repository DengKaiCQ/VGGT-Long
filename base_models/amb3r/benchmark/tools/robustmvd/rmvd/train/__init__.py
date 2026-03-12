# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/train/__init__.py

from .training import Training


def create_training(training_type, **kwargs):

    if training_type == 'mvd':
        from .multi_view_depth_training import MultiViewDepthTraining
        return MultiViewDepthTraining(**kwargs)


def list_trainings():
    return ['mvd']

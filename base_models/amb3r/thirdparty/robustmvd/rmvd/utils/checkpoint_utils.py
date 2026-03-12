# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/utils/checkpoint_utils.py

import os

import torch


class WeightsOnlySaver:
    def __init__(self, model=None, full_path=None, base_path=None, base_name=None, max_to_keep=None):

        assert full_path is not None or base_path is not None

        self.model = model
        self.full_path = full_path
        self.base_path = base_path
        self.base_name = base_name if base_name is not None else "snapshot"
        self.max_to_keep = max_to_keep

    def save(self, model=None, evo=None, epoch=None, iteration=None):

        assert model is not None or self.model is not None
        model = model if model is not None else self.model

        save_path = save_model(model=model, full_path=self.full_path, base_path=self.base_path,
                               base_name=self.base_name, evo=evo, epoch=epoch, iteration=iteration,
                               max_to_keep=self.max_to_keep)

        return save_path

    def get_checkpoints(self, include_iteration=False):

        if self.full_path is not None:
            checkpoints = get_checkpoints(self.full_path)
        else:
            checkpoints = get_checkpoints(path=self.base_path, base_name=self.base_name, include_iteration=False)

        if include_iteration:
            checkpoints = [(iteration_from_path(checkpoint), checkpoint) for checkpoint in checkpoints]

        return sorted(checkpoints)

    def get_latest_checkpoint(self, include_iteration=False):
        return self.get_checkpoints(include_iteration=include_iteration)[-1]

    def has_checkpoint(self, path=None):
        return path in self.get_checkpoints()

    def load(self, model=None, full_path=None, strict=True):

        assert model is not None or self.model is not None
        model = model if model is not None else self.model

        checkpoint = get_checkpoints(full_path)[-1] if full_path is not None else self.get_latest_checkpoint()
        print("Loading checkpoint {} (strict: {}).".format(checkpoint, strict))
        load_model(path=checkpoint, model=model, strict=strict)


class TrainStateSaver:
    def __init__(self, model=None, optim=None, scheduler=None, full_path=None, base_path=None, base_name=None,
                 max_to_keep=None):

        assert full_path is not None or base_path is not None

        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.full_path = full_path
        self.base_path = base_path
        self.base_name = base_name if base_name is not None else "snapshot"
        self.max_to_keep = max_to_keep

    def save(self, model=None, optim=None, scheduler=None, info_dict=None, evo=None, epoch=None, iteration=None):

        assert model is not None or self.model is not None
        assert optim is not None or self.optim is not None
        # it is okay if scheduler and self.scheduler are None
        model = model if model is not None else self.model
        optim = optim if optim is not None else self.optim
        scheduler = scheduler if scheduler is not None else self.scheduler

        save_path = save_all(model=model, optim=optim, scheduler=scheduler, info_dict=info_dict,
                             full_path=self.full_path, base_path=self.base_path, base_name=self.base_name,
                             evo=evo, epoch=epoch, iteration=iteration, max_to_keep=self.max_to_keep)

        return save_path

    def get_checkpoints(self, include_iteration=False):

        if self.full_path is not None:
            checkpoints = get_checkpoints(self.full_path)
        else:
            checkpoints = get_checkpoints(path=self.base_path, base_name=self.base_name, include_iteration=False)

        if include_iteration:
            checkpoints = [(iteration_from_path(checkpoint), checkpoint) for checkpoint in checkpoints]

        return sorted(checkpoints)

    def get_latest_checkpoint(self, include_iteration=False):
        return self.get_checkpoints(include_iteration=include_iteration)[-1]

    def has_checkpoint(self, path=None):
        return path in self.get_checkpoints()

    def load(self, model=None, optim=None, scheduler=None, full_path=None):

        assert model is not None or self.model is not None
        assert optim is not None or self.optim is not None
        # it is okay if scheduler and self.scheduler are None
        model = model if model is not None else self.model
        optim = optim if optim is not None else self.optim
        scheduler = scheduler if scheduler is not None else self.scheduler

        checkpoint = get_checkpoints(full_path)[-1] if full_path is not None else self.get_latest_checkpoint()
        print("Loading checkpoint {}).".format(checkpoint))
        out_dict = load_all(path=checkpoint, model=model, optim=optim, scheduler=scheduler)
        return out_dict


def is_checkpoint(path, base_name=None):

    file = os.path.split(path)[1]

    return os.path.isfile(path) and file.endswith(".pt") and (
        file.startswith(base_name) if base_name is not None else True)


def get_checkpoints(path, base_name=None, include_iteration=False):

    if os.path.isdir(path):
        checkpoints = [x for x in os.listdir(path)]
        checkpoints = [os.path.join(path, checkpoint) for checkpoint in checkpoints]
        checkpoints = [checkpoint for checkpoint in checkpoints if is_checkpoint(checkpoint, base_name)]
    elif os.path.isfile(path):
        checkpoints = [path] if is_checkpoint(path) else []
    else:
        checkpoints = []

    if include_iteration:
        checkpoints = [(iteration_from_path(checkpoint), checkpoint) for checkpoint in checkpoints]

    return checkpoints


def save_model(model, full_path=None, base_path=None, base_name=None, evo=None, epoch=None, iteration=None,
               max_to_keep=None):

    assert full_path is not None or base_path is not None

    if full_path is not None:
        path = full_path
        path = path + ".pt" if not (path.endswith(".pt") or path.endswith(".pth")) else path
    else:
        name = base_name if base_name is not None else "snapshot"
        name = name + "-evo-{:02d}".format(evo) if evo is not None else name
        name = name + "-epoch-{:04d}".format(epoch) if epoch is not None else name
        name = name + "-iter-{:09d}".format(iteration) if iteration is not None else name
        name += ".pt"
        path = os.path.join(base_path, name)

    torch.save(model.state_dict(), path)

    if full_path is None and max_to_keep is not None:
        base_name = base_name if base_name is not None else "snapshot"
        files = sorted([x for x in os.listdir(base_path) if x.startswith(base_name) and x.endswith(".pt")])

        while len(files) > max_to_keep:
            file_to_be_removed = files[0]
            os.remove(os.path.join(base_path, file_to_be_removed))
            del files[0]

    return path


def load_model(path, model, strict=True):
    model.load_state_dict(torch.load(path), strict=strict)


def save_all(model, optim, scheduler=None, info_dict=None,
             full_path=None, base_path=None, base_name=None, evo=None, epoch=None, iteration=None, max_to_keep=None):

    assert full_path is not None or base_path is not None

    if full_path is not None:
        path = full_path
        path = path + ".pt" if not (path.endswith(".pt") or path.endswith(".pth")) else path
    else:
        name = base_name if base_name is not None else "snapshot"
        name = name + "-evo-{:02d}".format(evo) if evo is not None else name
        name = name + "-epoch-{:04d}".format(epoch) if epoch is not None else name
        name = name + "-iter-{:09d}".format(iteration) if iteration is not None else name
        name += ".pt"
        path = os.path.join(base_path, name)

    torch.save(model.state_dict(), path)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if info_dict is not None:
        checkpoint.update(info_dict)

    torch.save(checkpoint, path)

    if full_path is None and max_to_keep is not None:
        base_name = base_name if base_name is not None else "snapshot"
        files = sorted([x for x in os.listdir(base_path) if x.startswith(base_name) and x.endswith(".pt")])

        while len(files) > max_to_keep:
            file_to_be_removed = files[0]
            os.remove(os.path.join(base_path, file_to_be_removed))
            del files[0]

    return path


def load_all(path, model, optim, scheduler=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    del checkpoint['model_state_dict']
    del checkpoint['optimizer_state_dict']

    return checkpoint


def iteration_from_path(path):
    idx = path.find('-iter-')
    iteration = int(path[idx + 6: idx + 6 + 9])
    return iteration

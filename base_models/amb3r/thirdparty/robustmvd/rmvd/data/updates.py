# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/data/updates.py

import os.path as osp
import abc
import pickle


class Update(metaclass=abc.ABCMeta):    
    def __init__(self):
        pass

    @abc.abstractmethod
    def load(self, orig_sample_dict, root=None):
        return


class Updates:
    def __init__(self, name, root=None, prefix=None, postfix=None, verbose=True, **kwargs):
        self.name = name
        self.verbose = verbose

        if self.verbose:
            print(f"Initializing dataset updates {self.name}.")

        self.root = root
        self.prefix = prefix
        self.postfix = postfix

        self.updates = {}
        self._init_updates(**kwargs)

        if self.verbose:
            if self.root is not None:
                print(f"\tData root: {self.root}")
            print(f"\tPrefix: {self.prefix}")
            print(f"\tPostfix: {self.postfix}")
            print(f"\tNumber of updates: {len(self)}")
            print(f"Finished initializing dataset update {self.name}.")

    @abc.abstractmethod
    def _init_updates(self):
        return

    def apply_update(self, sample_dict, index):
        if index in self:
            update = self[index]
            update_dict = update.load(orig_sample_dict=sample_dict, root=self.root)
            update_dict = self._add_pre_post_fixes(update_dict)
            sample_dict.update(update_dict)

    def _add_pre_post_fixes(self, update_dict):
        update_dict_out = {}

        for key, val in update_dict.items():
            key = (self.prefix if self.prefix is not None else "") + key + (self.postfix if self.postfix is not None else "")
            update_dict_out[key] = val

        return update_dict_out

    def __getitem__(self, index):
        return self.updates[index]

    def __contains__(self, index):
        return index in self.updates

    def __len__(self):
        return len(self.updates)

    # TODO: remove _init_updates and a classmethod from_file; pass updates=None as paramter to __init__ ?, remove PickledUpdates?


class PickledUpdates(Updates):

    def __init__(self, path, **kwargs):
        name = osp.splitext(osp.split(path)[1])[0]
        super().__init__(name=name, path=path, **kwargs)

    def _init_updates(self, path):
        with open(path, "rb") as file:
            self.updates = pickle.load(file)

    def write(self, path):
        with open(path, "wb") as file:
            pickle.dump(self.updates, file)

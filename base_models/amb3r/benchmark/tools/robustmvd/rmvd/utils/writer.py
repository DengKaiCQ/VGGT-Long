# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified 2023 by Philipp Schroeppel. Modifications include: removing things that are not required in rmvd
# and replace the visualization of numpy arrays.


# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/utils/writer.py

import os
import enum
from abc import abstractmethod
from time import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import make_np
import wandb

from .vis import vis, vis_image, vis_2d_array

EVENT_WRITERS = []
EVENT_STORAGE = []
GLOBAL_BUFFER = {}
MAX_BUFFER_SIZE = 20000


class EventType(enum.Enum):
    """Possible Event types and their associated write function"""

    TENSOR = "write_tensor"
    SCALAR = "write_scalar"
    HISTOGRAM = "write_histogram"


def put_scalar(name, scalar, step):
    """Setter function to place scalars into the queue to be written out

    Args:
        name: name of scalar
        scalar: value
        step: step associated with scalar
    """
    scalar = float(scalar)
    EVENT_STORAGE.append({"name": name, "write_type": EventType.SCALAR, "event": scalar, "step": step})


def put_scalar_dict(name, scalar, step):
    """Setter function to place a dictionary of scalars into the queue to be written out

    Args:
        name: name of scalar dictionary
        scalar: values to write out
        step: step associated with dict
    """
    for key, value in scalar.items():
        if isinstance(value, dict):
            put_scalar_dict(f"{name}/{key}", value, step)
        elif isinstance(value, list):
            put_scalar_list(f"{name}/{key}", value, step)
        else:
            put_scalar(f"{name}/{key}", value, step)


def put_scalar_list(name, scalar, step, every_nth=1, labels=None, max_to_log=None):
    """Setter function to place a list of scalars into the queue to be written out

    Args:
        name: name of scalar list
        scalar: values to write out
        step: step associated with list
        every_nth: write out every nth scalar
        labels: labels for each scalar
        max_to_log: maximum number of scalars to log
    """
    for scalar_idx, value in enumerate(scalar):
        if scalar_idx % every_nth == 0:
            if max_to_log is not None and scalar_idx >= max_to_log:
                break

            label = str(scalar_idx) if labels is None else labels[scalar_idx]

            if isinstance(value, dict):
                put_scalar_dict(f"{name}/{label}", value, step)
            elif isinstance(value, list):
                put_scalar_list(f"{name}/{label}", value, step)
            else:
                put_scalar(f"{name}/{label}", value, step)


def put_tensor(name, tensor, step, **kwargs):
    """Setter function to place tensors into the queue to be written out

    Args:
        name: name of tensor
        tensor: tensor to write out
        step: step associated with tensor
    """
    EVENT_STORAGE.append({"name": name, "write_type": EventType.TENSOR, "event": tensor, "step": step, "kwargs": kwargs})


def put_tensor_list(name, tensor, step, every_nth=1, labels=None, max_to_log=None, **kwargs):
    """Setter function to place a list of tensors into the queue to be written out

    Args:
        name: name of tensor list
        tensor: values to write out
        step: step associated with list
        every_nth: write out every nth tensor
        labels: labels for each tensor
        max_to_log: maximum number of tensors to log
    """
    for tensor_idx, value in enumerate(tensor):
        if tensor_idx % every_nth == 0:
            if max_to_log is not None and tensor_idx >= max_to_log:
                break

            label = str(tensor_idx) if labels is None else labels[tensor_idx]

            if isinstance(value, dict):
                put_tensor_dict(f"{name}/{label}", value, step, **kwargs)
            elif isinstance(value, list):
                put_tensor_list(f"{name}/{label}", value, step, **kwargs)
            else:
                put_tensor(f"{name}/{label}", value, step, **kwargs)


def put_tensor_dict(name, tensor, step, **kwargs):
    """Setter function to place a dictionary of scalars into the queue to be written out

    Args:
        name: name of scalar dictionary
        tensor: values to write out
        step: step associated with dict
    """
    for key, value in tensor.items():
        if isinstance(value, dict):
            put_tensor_dict(f"{name}/{key}", value, step, **kwargs)
        elif isinstance(value, list):
            put_tensor_list(f"{name}/{key}", value, step, **kwargs)
        else:
            put_tensor(f"{name}/{key}", value, step, **kwargs)


def put_histogram(name, values, step, replace_NaNs=False):
    """Setter function to place histogram into the queue to be written out

    Args:
        name: name of histogram
        values: values for histogram
        step: step associated with histogram
        replace_NaNs: indicates whether NaN values should be replaced with 0
    """

    values = make_np(values)

    if replace_NaNs:
        values = values.copy()
        values[~np.isfinite(values)] = 0

    values = values.reshape(-1)
    EVENT_STORAGE.append({"name": name, "write_type": EventType.HISTOGRAM, "event": values, "step": step})


def put_time(name, duration, step, write=True, avg_over_steps=True, update_eta=False):
    """Setter function to place a time element into the queue to be written out.
    Processes the time info according to the options.

    Args:
        name: name of time item
        duration: value
        step: step associated with value
        write: if True, log the time
        avg_over_steps: if True, calculate and record a running average of the times
        update_eta: if True, update the ETA. should only be set for the training iterations/s
    """

    if avg_over_steps:
        events = GLOBAL_BUFFER.get("events", {})
        GLOBAL_BUFFER["events"] = events
        curr_event = events.get(name, {"buffer": [], "avg": 0})
        curr_buffer = curr_event["buffer"]
        if len(curr_buffer) >= MAX_BUFFER_SIZE:
            curr_buffer.pop(0)
        curr_buffer.append(duration)
        curr_avg = sum(curr_buffer) / len(curr_buffer)
        if write:
            put_scalar(name, curr_avg, step)
        events[name] = {"buffer": curr_buffer, "avg": curr_avg}
    elif write:
        put_scalar(name, duration, step)

    if update_eta and write:
        remain_iter = GLOBAL_BUFFER["max_iter"] - step
        remain_time = remain_iter * GLOBAL_BUFFER["events"][name]["avg"]
        put_scalar("00_overview/eta_hours", remain_time/3600, step)
        put_scalar("00_overview/eta_days", remain_time/(3600*24), step)


def write_out_storage():
    """Function that writes all the events in storage to all the writer locations"""
    for writer in EVENT_WRITERS:
        for event in EVENT_STORAGE:
            write_func = getattr(writer, event["write_type"].value)
            if "kwargs" in event:
                write_func(event["name"], event["event"], event["step"], **event["kwargs"])
            else:
                write_func(event["name"], event["event"], event["step"])

    EVENT_STORAGE.clear()


def setup_writers(log_wandb, log_tensorboard, max_iterations, tensorboard_logs_dir=None, wandb_logs_dir=None, exp_id=None, config=None, comment=None):
    GLOBAL_BUFFER["max_iter"] = max_iterations

    if log_wandb:
        assert wandb_logs_dir is not None
        os.makedirs(wandb_logs_dir, exist_ok=True)
        curr_writer = WandbWriter(log_dir=wandb_logs_dir, exp_id=exp_id, config=config, comment=comment)
        EVENT_WRITERS.append(curr_writer)
    if log_tensorboard:
        assert tensorboard_logs_dir is not None
        os.makedirs(tensorboard_logs_dir, exist_ok=True)
        curr_writer = TensorboardWriter(log_dir=tensorboard_logs_dir)
        EVENT_WRITERS.append(curr_writer)


class Writer:
    """Writer class"""

    @abstractmethod
    def write_tensor(self, name, tensor, step, **kwargs):
        """method to write out tensor

        Args:
            name: data identifier
            tensor: rendered tensor to write
            step: the time step to log
        """
        raise NotImplementedError

    @abstractmethod
    def write_scalar(self, name, scalar, step, **kwargs):
        """Required method to write a single scalar value to the logger

        Args:
            name: data identifier
            scalar: value to write out
            step: the time step to log
        """
        raise NotImplementedError


class TimeWriter:
    """Timer context manager that calculates duration around wrapped functions"""

    def __init__(self, name, step, write=True, avg_over_steps=False, update_eta=False):
        self.name = name
        self.step = step
        self.write = write
        self.avg_over_steps = avg_over_steps
        self.update_eta = update_eta

        self.start: float = 0.0
        self.duration: float = 0.0

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        self.duration = time() - self.start
        put_time(
            name=self.name,
            duration=self.duration,
            step=self.step,
            write=self.write,
            avg_over_steps=self.avg_over_steps,
            update_eta=self.update_eta,
        )


class WandbWriter(Writer):
    """WandDB Writer Class"""

    def __init__(self, log_dir, exp_id=None, config=None, comment=None):
        wandb.init(project="robustmvd", dir=str(log_dir), reinit=True, resume=True, config=config, name=exp_id, notes=comment)

    def write_tensor(self, name, tensor, step, **kwargs):
        img = vis(tensor, **kwargs)
        wandb.log({name: wandb.Image(img)}, step=step)

    def write_scalar(self, name, scalar, step):
        scalar = float(scalar)
        wandb.log({name: scalar}, step=step)

    def write_histogram(self, name, values, step):
        wandb.log({name: wandb.Histogram(values)}, step=step)


class TensorboardWriter(Writer):
    """Tensorboard Writer Class"""

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def write_tensor(self, name, tensor, step, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "out_format"}
        img = vis(tensor, out_format={'type': 'np', 'mode': 'RGB', 'dtype': 'uint8'}, **filtered_kwargs)
        self.writer.add_image(name, img, step)

    def write_scalar(self, name, scalar, step):
        scalar = float(scalar)
        self.writer.add_scalar(name, scalar, step)

    def write_histogram(self, name, values, step):
        self.writer.add_histogram(name, values, step)


def _format_time(seconds):
    """utility tool to format time in human-readable form given seconds"""
    ms = seconds % 1
    ms = ms * 1e3
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return f"{days} d, {hours} h, {minutes} m, {seconds} s"
    if hours > 0:
        return f"{hours} h, {minutes} m, {seconds} s"
    if minutes > 0:
        return f"{minutes} m, {seconds} s"
    if seconds > 0:
        return f"{seconds} s, {ms:0.3f} ms"

    return f"{ms:0.3f} ms"

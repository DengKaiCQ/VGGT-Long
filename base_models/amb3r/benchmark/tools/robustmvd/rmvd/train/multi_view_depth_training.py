# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/train/multi_view_depth_training.py

import os
import os.path as osp
from typing import Optional, Sequence
import time

import torch

from rmvd import create_batch_augmentation
from rmvd.utils import TrainStateSaver, WeightsOnlySaver, to_torch, get_torch_model_device, \
    count_torch_model_parameters, writer, select_by_index, exclude_index

# TODO: add proper logger that writes to file + optionally outputs timestamps


class MultiViewDepthTraining:
    """Multi-view depth training.

    Can be applied to a dataset and model to train the model on the dataset.

    Args:
        out_dir: Directory where results will be written. If None, results are not written to disk.
        model: Model to train.
        dataset: Dataset for training.
        optimizer: Optimizer for training.
        scheduler: Scheduler for training.
        loss: Loss function for training.
        batch_size: Batch size for training.
        max_iterations: Maximum number of iterations to train.
        inputs: List of input modalities that are supplied to the model.
            Can include: ["images", "intrinsics", "poses", "depth_range"].
            Default: ["images", "intrinsics", "poses"].
        batch_augmentations: List of augmentations to apply to each batch jointly. Default: do not apply any batch augmentations.
        alignment: Alignment between predicted and ground truth depths. Options are None, "median",
            "least_squares_scale_shift".
            None evaluates predictions without any alignment.
            "median" scales predicted depth maps with the ratio of medians of predicted and ground truth depth maps.
            "least_squares_scale_shift" scales and shifts predicted depth maps such that the least-squared error to the
            ground truth is minimal.
        grad_clip_max_norm: Maximum norm for gradient clipping. Default: None.
        num_workers: Number of workers for the dataloader. Default: 8.
        print_interval: Interval for printing training state. Default: 100.
        log_loss_interval: Interval for logging loss. Default: 100 iterations.
        log_interval: Interval for logging training state. Default: 5000 iterations.
        save_checkpoint_interval_min: Interval in minutes for saving checkpoints. Default: 20 minutes.
        log_tensorboard: Whether to log to tensorboard. Default: True.
        log_wandb: Whether to log to wandb. Default: False.
        log_full_batch: Whether to log the full batch. Default: False.
        verbose: Print training details.
    """

    def __init__(self,
                 out_dir: str,
                 model: torch.nn.Module,
                 dataset: torch.utils.data.Dataset,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 loss: torch.nn.Module,
                 batch_size: int,
                 max_iterations: int,
                 inputs: Sequence[str] = None,
                 batch_augmentations: Optional[Sequence[str]] = None,
                 alignment: Optional[str] = None,
                 grad_clip_max_norm: Optional[float] = None,
                 num_workers: Optional[int] = 8,
                 print_interval: Optional[int] = 100,
                 log_loss_interval: Optional[int] = 100,
                 log_interval: Optional[int] = 5000,
                 save_checkpoint_interval_min: Optional[int] = 20,
                 log_full_batch: Optional[bool] = False,
                 verbose: bool = True,
                 **_, ):


        # TODO: set up logging

        self.verbose = verbose

        if self.verbose:
            print(f"Initializing training {self.name}.")

        self.out_dir = out_dir
        self._init_dirs()

        # will be set/used in __call__:

        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.batch_size = batch_size
        self.grad_clip_max_norm = grad_clip_max_norm
        self.dataloader = self.dataset.get_loader(batch_size=batch_size, shuffle=True, pin_memory=True,
                                                  num_workers=num_workers, drop_last=True)

        self.max_iterations = max_iterations
        self.finished_iterations = 0
        self.saver_all = None
        self.saver_weights_only = None
        self._start_iteration = None
        self._setup_savers()
        self._restore_weights()
        self._start_iteration = self.finished_iterations

        self.inputs = list(set(inputs + ["images"])) if inputs is not None else ["images", "intrinsics", "poses"]
        
        batch_augmentations = [] if batch_augmentations is None else batch_augmentations
        batch_augmentations = [batch_augmentations] if not isinstance(batch_augmentations, list) else batch_augmentations
        self.batch_augmentations = []
        self._init_batch_augmentations(batch_augmentations)
        
        self.alignment = alignment
        assert self.alignment is None, "Alignment is not yet implemented."

        self.log_full_batch = log_full_batch

        self.print_interval = print_interval
        self.log_interval = log_interval
        self.log_loss_interval = log_loss_interval
        self.save_checkpoint_interval_min = save_checkpoint_interval_min

        if self.verbose:
            print(self)
            print(f"Finished initializing training {self.name}.")
            print()

    @property
    def name(self):
        return type(self).__name__

    def __str__(self):
        ret = f"{self.name} with settings:"
        ret += f"\n\tOutput directory: {self.out_dir}"
        ret += f"\n\tModel: {self.model.name}"
        ret += f"\n\tModel parameter count: {count_torch_model_parameters(self.model)}"
        ret += f"\n\tDataset: {self.dataset.name}"
        ret += f"\n\tDataset size: {len(self.dataset)}"
        ret += f"\n\tOptimizer: {self.optimizer.name}"
        ret += f"\n\tScheduler: {self.scheduler.name}"
        ret += f"\n\tGrad clip max norm: {self.grad_clip_max_norm}"
        ret += f"\n\tLoss: {self.loss.name}"
        ret += f"\n\tBatch size: {self.batch_size}"
        ret += f"\n\tInputs: {self.inputs}"
        ret += f"\n\tAlignment: {self.alignment}"
        ret += f"\n\tFinished iterations: {self.finished_iterations}"
        ret += f"\n\tMax iterations: {self.max_iterations}"  # TODO: add num epochs = max_iterations / len(dataset)
        return ret

    def _init_dirs(self):
        self.log_file_path = osp.join(self.out_dir, "log.txt")
        self.artifacts_dir = osp.join(self.out_dir, "artifacts")
        self.checkpoints_dir = osp.join(self.out_dir, "checkpoints")
        self.weights_only_checkpoints_dir = osp.join(self.out_dir, "weights_only_checkpoints_dir")
        self.checkpoints_name = "snapshot"
        self.weights_only_checkpoints_name = "snapshot"

        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.weights_only_checkpoints_dir, exist_ok=True)

        # TODO: log.add_log_file(self.log_file_path, flush_line=True)

    def _init_batch_augmentations(self, batch_augmentations):
        for batch_augmentation in batch_augmentations:
            if isinstance(batch_augmentation, str):
                batch_augmentation = create_batch_augmentation(batch_augmentation)
            self.batch_augmentations.append(batch_augmentation)

    def __call__(self):
        """Run training."""

        should_continue = lambda: self.finished_iterations < self.max_iterations

        if not should_continue():
            print("Training already finished.")
            return

        if self.verbose:
            print(f"Starting training {self.name}.")

        self.model.train()

        should_print = lambda: self.finished_iterations % self.print_interval == 0
        should_log = lambda: self.finished_iterations % self.log_interval == 0
        should_log_loss = lambda: self.finished_iterations % self.log_loss_interval == 0

        steps_since_print = 0
        start_print_interval = time.time()
        last_checkpoint_time = time.time()

        while should_continue():
            for iter_in_epoch, sample in enumerate(self.dataloader):
                with writer.TimeWriter(name="00_overview/train_sec_iter", step=self.finished_iterations, write=should_log_loss(), avg_over_steps=True, update_eta=True):
                    self.optimizer.zero_grad()

                    for batch_augmentation in self.batch_augmentations:
                        batch_augmentation(sample)
                    sample = to_torch(sample, device=get_torch_model_device(self.model))
                    sample_inputs, sample_gt = self._inputs_and_gt_from_sample(sample)

                    pred, aux = self.model(**sample_inputs)

                    loss, sub_losses, pointwise_losses = self.loss(sample_inputs=sample_inputs, sample_gt=sample_gt,
                                                                   pred=pred, aux=aux, iteration=self.finished_iterations)
                    loss.backward()
                    if self.grad_clip_max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)

                    self.optimizer.step()
                    self.scheduler.step()

                steps_since_print += 1
                if should_print():
                    end_print_interval = time.time()
                    time_per_iteration = (end_print_interval - start_print_interval) / steps_since_print
                    print(f"Iteration {self.finished_iterations}/{self.max_iterations} - "
                          f"{time_per_iteration:1.4f} sec per iteration - loss: {loss:1.5f}")
                    start_print_interval = time.time()
                    steps_since_print = 0

                if should_log():
                    self._log_all(sample_inputs, sample_gt, pred, aux, loss, sub_losses, pointwise_losses)
                elif should_log_loss():
                    self._log_loss(loss, sub_losses, pointwise_losses)

                self.finished_iterations += 1

                if self._start_iteration < self.finished_iterations < self.max_iterations:
                    if time.time() - last_checkpoint_time > 60 * self.save_checkpoint_interval_min:
                        self._save_all()
                        last_checkpoint_time = time.time()

                writer.write_out_storage()

                if not should_continue():
                    break

        self._write_checkpoints()

        if self.verbose:
            print(f"Finished training {self.name}.")

    def _inputs_and_gt_from_sample(self, sample):
        is_input = lambda key: key in self.inputs or key == "keyview_idx"
        sample_inputs = {key: val for key, val in sample.items() if is_input(key)}
        sample_gt = {key: val for key, val in sample.items() if not is_input(key)}
        return sample_inputs, sample_gt

    def _setup_savers(self):
        max_checkpoints_to_keep = 3
        self.saver_all = TrainStateSaver(model=self.model, optim=self.optimizer,
                                         scheduler=self.scheduler,
                                         base_path=self.checkpoints_dir,
                                         base_name=self.checkpoints_name,
                                         max_to_keep=max_checkpoints_to_keep)

        self.saver_weights_only = WeightsOnlySaver(model=self.model,
                                                   base_path=self.weights_only_checkpoints_dir,
                                                   base_name=self.weights_only_checkpoints_name)

    def _get_all_checkpoints(self):
        checkpoints = self.saver_all.get_checkpoints(include_iteration=True) + self.saver_weights_only.get_checkpoints(
            include_iteration=True)
        return sorted(checkpoints)

    def _restore_weights(self):
        all_checkpoints = self._get_all_checkpoints()
        if len(all_checkpoints) > 0:
            
            print("Existing checkpoints:")
            for step, checkpoint in all_checkpoints:
                print(f"\t{step}: {checkpoint}")
                
            newest_checkpoint = all_checkpoints[-1][1]
            if self.saver_all.has_checkpoint(newest_checkpoint):
                print(f"Restoring training state from checkpoint {newest_checkpoint}.")
                self.saver_all.load(full_path=newest_checkpoint)
                
            elif self.saver_weights_only.has_checkpoint(newest_checkpoint):
                print(f"Restoring weights-only from checkpoint {newest_checkpoint}.")
                self.saver_weights_only.load(full_path=newest_checkpoint, strict=True)
                
            else:
                raise ValueError(f"Checkpoint {newest_checkpoint} does not belong to any of the known savers.")

            self.finished_iterations = all_checkpoints[-1][0]

    def _write_checkpoints(self):
        if self.finished_iterations > self._start_iteration:
            self._save_all()

            if self.finished_iterations >= self.max_iterations:
                self._save_weights_only()

    def _save_all(self):
        save_path = self.saver_all.save(iteration=self.finished_iterations)
        print(f"Saved training state checkpoint to {save_path}.")

    def _save_weights_only(self):
        save_path = self.saver_weights_only.save(iteration=self.finished_iterations)
        print(f"Saved weights-only checkpoint to {save_path}.")

    def _log_all(self, sample_inputs, sample_gt, pred, aux, loss, sub_losses, pointwise_losses):
        self._log_in_data(sample_inputs, sample_gt)
        self._log_pred(pred, aux)
        self._log_loss(loss, sub_losses, pointwise_losses, scalars_only=False)
        self._log_optim()

    def _log_in_data(self, sample_inputs, sample_gt):
        base_name = '01_in'
        images = sample_inputs['images']
        keyview_idx = sample_inputs['keyview_idx']

        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)

        writer.put_tensor(name=f"{base_name}/00_key_image", tensor=image_key, step=self.finished_iterations
                          , full_batch=self.log_full_batch)
        writer.put_tensor_list(name=f"{base_name}/01_source_images", tensor=images_source,
                               step=self.finished_iterations, full_batch=self.log_full_batch)

        gt_depth = sample_gt['depth']
        gt_invdepth = sample_gt['invdepth']
        writer.put_tensor(name=f"{base_name}/02_gt_depth", tensor=gt_depth, step=self.finished_iterations,
                          full_batch=self.log_full_batch, invalid_values=[0], mark_invalid=True)
        writer.put_tensor(name=f"{base_name}/03_gt_invdepth", tensor=gt_invdepth, step=self.finished_iterations,
                          full_batch=self.log_full_batch, invalid_values=[0], mark_invalid=True)

    def _log_pred(self, pred, aux):
        base_name = '02_pred'
        pred_depth = pred['depth']
        writer.put_tensor(name=f"{base_name}/00_pred_depth", tensor=pred_depth, step=self.finished_iterations,
                          full_batch=self.log_full_batch)

        if 'depth_uncertainty' in pred:
            pred_depth_uncertainty = pred['depth_uncertainty']
            writer.put_tensor(name=f"{base_name}/01_pred_depth_uncertainty", tensor=pred_depth_uncertainty,
                              step=self.finished_iterations, full_batch=self.log_full_batch)

        writer.put_tensor_dict(name=f"{base_name}/02_aux", tensor=aux, step=self.finished_iterations,
                               full_batch=self.log_full_batch)

    def _log_loss(self, loss, sub_losses, pointwise_losses, scalars_only=True):
        base_name = '03_loss'
        writer.put_scalar(name="00_overview/00_loss", scalar=loss, step=self.finished_iterations)
        writer.put_scalar(name=f"{base_name}/00_total_loss", scalar=loss, step=self.finished_iterations)

        writer.put_scalar_dict(name=f"{base_name}/01_sub", scalar=sub_losses, step=self.finished_iterations)

        if not scalars_only:
            writer.put_tensor_dict(name=f"{base_name}/02_qual", tensor=pointwise_losses, step=self.finished_iterations)

    def _log_optim(self):
        base_name = "04_optim"
        optimizer = self.optimizer
        model = self.model

        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group["lr"]
            writer.put_scalar(name=f"00_overview/lr/group_{i}", scalar=lr, step=self.finished_iterations)

        for name, param in model.named_parameters():
            writer.put_histogram(name=f"{base_name}/0_vals/{name}", values=param, step=self.finished_iterations)

            if param.grad is not None:
                writer.put_histogram(name=f"{base_name}/1_grads/{name}", values=param.grad,
                                     step=self.finished_iterations)

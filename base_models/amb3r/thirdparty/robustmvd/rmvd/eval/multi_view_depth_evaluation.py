# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/eval/multi_view_depth_evaluation.py

import os
import os.path as osp
import time
from copy import deepcopy
from typing import Optional, Sequence, Tuple, Union
import warnings
import pickle

import torch
import skimage.transform
import numpy as np
import pandas as pd

from rmvd.utils import numpy_collate, vis, select_by_index, get_full_class_name
from rmvd.data.layout import Layout, Visualization
from .metrics import m_rel_ae, pointwise_rel_ae, thresh_inliers, sparsification
from rmvd.data.updates import Update


class MultiViewDepthEvaluation:
    """Multi-view depth evaluation.

    Can be applied to a dataset and model to evaluate the depth estimation performance of the model on the dataset.
    Supports multiple input modalities and an optional alignment between predicted and GT depths.
    Can additionally evaluate the estimated uncertainty of the model if available.

    A typical MVS model would set
        inputs=["images", "intrinsics", "poses", "depth_range"], alignment=None.

    A typical depth-from-video model would set
        inputs=["images", "intrinsics"], alignment="median".

    A typical depth-from-single-view model would set
        inputs=["images"], max_source_views=0, alignment="median".

    Args:
        out_dir: Directory where results will be written. If None, results are not written to disk.
        inputs: List of input modalities that are supplied to the model.
            Can include: ["images", "intrinsics", "poses", "depth_range"].
            Default: ["images"]
        alignment: Alignment between predicted and ground truth depths. Options are None, "median", "translation".
            None evaluates predictions without any alignment.
            "median" scales predicted depth maps with the ratio of medians of predicted and ground truth depth maps.
            "translation" scales predicted depth maps with the ratio of the predicted and ground truth translation.
        max_source_views: Maximum number of source views to be considered. None means all available source views are
            considered. Default: None.
        min_source_views. Minimum number of source views provided to the model.
            If max_source_views is not None, is set to min(min_source_views, max_source_views). Default: 1.
        view_ordering: Ordering of source views during the evaluation.
            Options are "quasi-optimal" and "nearest". Default: "quasi-optimal".
            "quasi-optimal": evaluate predicted depth maps for all (keyview, sourceview) pairs.
                Order source views according to the prediction accuracy. Increase source view set based on
                the obtained ordering and re-evaluate for each additional source view.
                Log results based on the number of source views. Log best results as overall results.
            "nearest": evaluate predicted depth maps for increasing number of source views. Increase source
                view set based on the ordering of views in the sample, i.e. based on the distance between source
                view indices and the keyview index. Log results based on the number of source views.
                Log best results as overall results.
        eval_uncertainty: Evaluate predicted uncertainty (pred_depth_uncertainty) if available.
            Increases evaluation time.
        clip_pred_depth: Clip model predictions before evaluation to a reasonable range. This makes sense to reduce
            the effect of unreasonable outliers. Defaults to True.
        sparse_pred: Predicted depth is sparse. Invalid predictions are indicated by 0 values and ignored in
            the evaluation. Defaults to True.
        verbose: Print evaluation details.
    """
    def __init__(self,
                 out_dir: Optional[str] = None,
                 inputs: Sequence[str] = None,
                 alignment: Optional[str] = None,
                 max_source_views: Optional[int] = None,
                 min_source_views: int = 1,
                 view_ordering: str = "quasi-optimal",
                 eval_uncertainty: bool = True,
                 clip_pred_depth: Union[bool, Tuple[float, float]] = True,
                 sparse_pred: bool = False,
                 verbose: bool = True,
                 **_
                 ):

        self.verbose = verbose

        if self.verbose:
            print(f"Initializing evaluation {self.name}.")

        self.out_dir = out_dir
        if self.out_dir is not None:
            self.quantitatives_dir = osp.join(self.out_dir, "quantitative")
            self.sample_results_dir = osp.join(self.quantitatives_dir, "per_sample")
            self.qualitatives_dir = osp.join(self.out_dir, "qualitative")
            os.makedirs(self.out_dir, exist_ok=True)
            os.makedirs(self.quantitatives_dir, exist_ok=True)
            os.makedirs(self.sample_results_dir, exist_ok=True)
            os.makedirs(self.qualitatives_dir, exist_ok=True)

        self.inputs = list(set(inputs + ["images"])) if inputs is not None else ["images"]
        self.alignment = alignment
        self.max_source_views = max_source_views
        self.min_source_views = min_source_views if max_source_views is None else min(min_source_views, max_source_views)
        self.view_ordering = view_ordering if (self.max_source_views is None) or (self.max_source_views > 0) else None
        self.eval_uncertainty = eval_uncertainty
        self.clip_pred_depth = clip_pred_depth
        self.sparse_pred = sparse_pred

        # will be set/used in __call__:
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.exp_name = None
        self.sample_indices = None
        self.qualitative_indices = None
        self.burn_in_samples = None
        self.cur_sample_num = 0
        self.cur_sample_idx = 0
        self.results = None
        self.sparsification_curves = None
        self.dataset_updates = None

        if self.verbose:
            print(self)
            print(f"Finished initializing evaluation {self.name}.")
            print()

    @property
    def name(self):
        return type(self).__name__

    def __str__(self):
        ret = f"{self.name} with settings:"
        ret += f"\n\tInputs: {self.inputs}"
        ret += f"\n\tAlignment: {self.alignment}"
        ret += f"\n\tMin source views: {self.min_source_views}"
        ret += f"\n\tMax source views: {'All' if self.max_source_views is None else self.max_source_views}"
        ret += f"\n\tView ordering: {self.view_ordering}"
        ret += f"\n\tEvaluate uncertainty: {self.eval_uncertainty}"
        ret += f"\n\tClip predicted depth: {self.clip_pred_depth}"
        ret += f"\n\tPredicted depth is sparse: {self.sparse_pred}"
        if self.out_dir is not None:
            ret += f"\n\tOutput directory: {self.out_dir}"
        else:
            ret += "\n\tOutput directory: None. Results will not be written to disk!"
        return ret

    @torch.no_grad()
    def __call__(self,
                 dataset,
                 model,
                 samples: Optional[Union[int, Sequence[int]]] = None,
                 qualitatives: Union[int, Sequence[int]] = 10,
                 burn_in_samples: int = 3,
                 exp_name: Optional[str] = None,
                 **_):
        """Run depth evaluation for a dataset and model.

        Args:
            dataset: Dataset for the evaluation.
            model: Class or function that applies the model to a sample.
            samples: Integer that indicates the number of samples that should be evaluated or list that indicates
                the indices of samples that should be evaluated. None evaluates all samples.
            qualitatives: Integer that indicates the number of qualitatives that should be logged or list that indicates
                the indices of samples for which qualitatives should be logged. -1 logs qualitatives for all samples.
            burn_in_samples: Number of samples that will not be considered for runtime/memory measurements.
                Defaults to 3.
            exp_name: Name of the experiment. Optional.

        Returns:
            Results of the evaluation.
        """
        self._init_evaluation(dataset=dataset, model=model, samples=samples, qualitatives=qualitatives,
                              burn_in_samples=burn_in_samples, exp_name=exp_name)
        results = self._evaluate()
        self._output_results()
        self._reset_evaluation()
        return results

    def _init_evaluation(self,
                         dataset,
                         model,
                         samples=None,
                         qualitatives=10,
                         burn_in_samples=3,
                         exp_name=None,):
        self.dataset = dataset
        self.model = model
        self.exp_name = exp_name
        self._init_sample_indices(samples=samples)
        self._init_qualitative_indices(qualitatives=qualitatives)
        self._init_results()
        self.burn_in_samples = burn_in_samples
        self.dataloader = self.dataset.get_loader(batch_size=1, indices=self.sample_indices, num_workers=0,
                                                  collate_fn=numpy_collate)
        # we fix a batch_size=1 to ensure comparable runtimes

    def _init_sample_indices(self, samples):
        if isinstance(samples, list):
            self.sample_indices = samples
            if self.verbose:
                print(f"Evaluating samples with indices: {self.sample_indices}.")
        elif isinstance(samples, int) and samples > 0:
            step_size = len(self.dataset) / samples  # <=1
            self.sample_indices = [int(i*step_size) for i in range(samples)]
            if self.verbose:
                print(f"Evaluating samples with indices: {self.sample_indices}.")
        else:
            self.sample_indices = list(range(len(self.dataset)))

    def _init_qualitative_indices(self, qualitatives=None):
        if qualitatives is None:
            self.qualitative_indices = []
        elif isinstance(qualitatives, list):
            self.qualitative_indices = qualitatives
        elif isinstance(qualitatives, int):
            if qualitatives < 0:
                self.qualitative_indices = self.sample_indices
            else:
                step_size = len(self.sample_indices) / qualitatives  # <=1
                self.qualitative_indices = list(
                    set([self.sample_indices[int(i * step_size)] for i in range(qualitatives)]))

    def _evaluate(self):
        for sample_num, (sample_idx, sample) in enumerate(zip(self.sample_indices, self.dataloader)):
            self.cur_sample_num = sample_num
            self.cur_sample_idx = sample_idx

            if self.verbose:
                print(f"Processing sample {self.cur_sample_num+1} / {len(self.sample_indices)} "
                      f"(index: {self.cur_sample_idx}):")

            # prepare sample:
            should_qualitative = (self.cur_sample_idx in self.qualitative_indices) and (self.out_dir is not None)
            keyview_idx = int(sample['keyview_idx'])
            sample_inputs, sample_gt = self._inputs_and_gt_from_sample(sample)

            # get evaluation order:
            ordered_source_indices = self._get_source_view_ordering(sample_inputs=sample_inputs, sample_gt=sample_gt)

            max_source_views = min(len(ordered_source_indices), self.max_source_views) \
                if self.max_source_views is not None else len(ordered_source_indices)
            min_source_views = self.min_source_views

            best_metrics = None
            best_pred = None

            for num_source_views in range(min_source_views, max_source_views+1):
                cur_source_indices = ordered_source_indices[:num_source_views]
                cur_view_indices = list(sorted([keyview_idx] + cur_source_indices))

                if self.verbose:
                    print(f"\tEvaluating with {num_source_views} / {max_source_views} source views:  {cur_source_indices}")

                self._reset_memory_stats()

                # construct current input sample:
                cur_sample_gt = deepcopy(sample_gt)
                cur_sample_inputs = filter_views_in_sample(sample=sample_inputs, indices_to_keep=cur_view_indices)

                # run model:
                pred, runtimes, gpu_mem = self._run_model(cur_sample_inputs)
                self._postprocess_sample_and_output(cur_sample_inputs, cur_sample_gt, pred)

                # compute and log metrics:
                metrics = self._compute_metrics(sample_inputs=cur_sample_inputs,
                                                sample_gt=cur_sample_gt,
                                                pred=pred)
                metrics.update(runtimes)
                metrics.update(gpu_mem)
                self._log_metrics(metrics, num_source_views)

                if self.verbose:
                    if self.alignment == None:
                        print(f"\t\tAbsrel={metrics['absrel']}, inliers103={metrics['inliers103']}, inliers125={metrics['inliers125']}")
                    else:
                        print(f"\t\tAbsrel={metrics['absrel']}, inliers103={metrics['inliers103']}")

                if np.isfinite(metrics['absrel']) and \
                        (best_metrics is None or metrics['absrel'] < best_metrics['absrel']):
                    best_metrics = metrics
                    best_metrics['num_views'] = num_source_views
                    best_pred = pred

            if self.eval_uncertainty:
                uncertainty_metrics = self._compute_uncertainty_metrics(sample_inputs=cur_sample_inputs,
                                                                        sample_gt=cur_sample_gt,
                                                                        pred=best_pred)
                best_metrics.update(uncertainty_metrics)

            # log best metrics for this sample:
            self._log_metrics(best_metrics, 'best')

            # compute and log qualitatives:
            if should_qualitative:
                qualitatives = self._compute_qualitatives(sample_inputs=sample_inputs, sample_gt=sample_gt,
                                                          pred=best_pred)
                self._log_qualitatives(qualitatives)
                # write dataset update only for samples where we computed qualitatives:
                self._add_dataset_update(best_metrics)

            if self.verbose:
                if self.alignment == None:
                    print(f"Sample with index {self.cur_sample_idx} has AbsRel={best_metrics['absrel']} "
                        f"with {best_metrics['num_views']} source views, running average of best AbsRel is "
                        f"{self.results['best']['absrel'].mean()}, inliers103 is "
                        f"{self.results['best']['inliers103'].mean()}, inliers125 is "
                        f"{self.results['best']['inliers125'].mean()}.\n")
                else:
                    print(f"Sample with index {self.cur_sample_idx} has AbsRel={best_metrics['absrel']} "
                        f"with {best_metrics['num_views']} source views, running average of best AbsRel is "
                        f"{self.results['best']['absrel'].mean()}, inliers103 is "
                        f"{self.results['best']['inliers103'].mean()}.\n")

        return self.results

    def _compute_qualitatives(self, sample_inputs, sample_gt, pred):

        gt_depth = sample_gt['depth'][0]

        pred_depth = pred['depth'][0]
        pred_invdepth = pred['invdepth'][0]
        pred_mask = pred_depth != 0 if self.sparse_pred else np.ones_like(pred_depth, dtype=bool)

        pointwise_absrel = pointwise_rel_ae(gt=gt_depth, pred=pred_depth, mask=pred_mask)

        qualitatives = {'pointwise_absrel': pointwise_absrel,
                        'pred_depth': pred_depth,
                        'pred_invdepth': pred_invdepth}

        if 'depth_uncertainty' in pred:
            qualitatives['pred_depth_uncertainty'] = pred['depth_uncertainty'][0]

        return qualitatives

    def _log_qualitatives(self, qualitatives):

        for qualitative_name, qualitative in qualitatives.items():
            out_path = osp.join(self.qualitatives_dir, f'{self.cur_sample_idx:07d}-{qualitative_name}')
            npy_path = out_path + '.npy'
            png_path = out_path + '.png'
            np.save(npy_path, qualitative)
            vis(qualitative).save(png_path)

            self._add_dataset_update({qualitative_name: npy_path})

    def _add_dataset_update(self, update_dict):
        if self.cur_sample_idx not in self.dataset_updates:
            self.dataset_updates[self.cur_sample_idx] = MultiMultiViewDepthEvaluationUpdate()

        self.dataset_updates[self.cur_sample_idx].update_dict.update(update_dict)

    def _reset_evaluation(self):
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.exp_name = None
        self.sample_indices = None
        self.qualitative_indices = None
        self.burn_in_samples = None
        self.cur_sample_idx = 0
        self.cur_sample_num = 0
        self.results = None
        self.sparsification_curves = None
        self.dataset_updates = None

    def _init_results(self):
        self.results = pd.DataFrame()
        self.results.index.name = 'sample_idx'
        self.results.columns.name = 'metric'
        self.results = pd.concat({1: self.results}, axis=1, names=["num_views"])

        if self.eval_uncertainty:
            x = np.linspace(0, 0.99, 100)
            columns = pd.Index(x, name="frac_removed")
            index = pd.MultiIndex.from_tuples([], names=("sample_idx", "curve"))
            self.sparsification_curves = pd.DataFrame(columns=columns, index=index)
            
        self.dataset_updates = {}

    def _get_source_view_ordering(self, sample_inputs, sample_gt):
        if self.view_ordering == 'quasi-optimal':
            return self._get_quasi_optimal_source_view_ordering(sample_inputs=sample_inputs, sample_gt=sample_gt)
        elif (self.view_ordering == 'nearest') or (self.view_ordering is None):
            return self._get_nearest_source_view_ordering(sample_inputs=sample_inputs, sample_gt=sample_gt)

    def _get_nearest_source_view_ordering(self, sample_inputs, sample_gt):
        keyview_idx = int(sample_inputs['keyview_idx'])
        source_indices = [idx for idx in range(len(sample_inputs['images'])) if idx != keyview_idx]
        source_view_ordering = sorted(source_indices, key=lambda x: np.abs(x-keyview_idx))
        return source_view_ordering

    def _get_quasi_optimal_source_view_ordering(self, sample_inputs, sample_gt):
        keyview_idx = int(sample_inputs['keyview_idx'])
        source_indices = [idx for idx in range(len(sample_inputs['images'])) if idx != keyview_idx]
        source_scores = {}

        for source_idx in source_indices:
            # construct temporary sample with a single source view:
            indices_to_keep = [keyview_idx, source_idx]
            cur_sample_gt = deepcopy(sample_gt)
            cur_sample_inputs = filter_views_in_sample(sample=sample_inputs, indices_to_keep=indices_to_keep)

            # run model:
            pred, _, _ = self._run_model(cur_sample_inputs)
            self._postprocess_sample_and_output(cur_sample_inputs, cur_sample_gt, pred)

            # compute absrel for using current source view:
            metrics = self._compute_metrics(sample_inputs=cur_sample_inputs,
                                            sample_gt=cur_sample_gt, pred=pred)
            source_scores[source_idx] = metrics['absrel']

        source_view_ordering = sorted(source_scores, key=source_scores.get)
        return source_view_ordering

    def _reset_memory_stats(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

    def _inputs_and_gt_from_sample(self, sample):
        is_input = lambda key: key in self.inputs or key == "keyview_idx"
        sample_inputs = {key: val for key, val in sample.items() if is_input(key)}
        sample_gt = {key: val for key, val in sample.items() if not is_input(key)}
        return sample_inputs, sample_gt

    def _postprocess_sample_and_output(self, sample_inputs, sample_gt, pred):

        gt_depth = sample_gt['depth']

        pred_depth = pred['depth']
        pred_depth = skimage.transform.resize(pred_depth, gt_depth.shape, order=0, anti_aliasing=False)

        pred_mask = pred_depth != 0 if self.sparse_pred else np.ones_like(pred_depth, dtype=bool)
        gt_mask = gt_depth > 0

        if self.alignment == "median":
            mask = gt_mask & pred_mask
            ratio = np.median(gt_depth[mask]) / np.median(pred_depth[mask])

            if mask.any() and np.isfinite(ratio):
                pred_depth = pred_depth * ratio
            else:
                ratio = np.nan

            pred['scaling_factor'] = ratio

        elif self.alignment == 'least_squares_scale_shift':
            mask = gt_mask & pred_mask
            with np.errstate(divide='ignore', invalid='ignore'):
                pred_invdepth = np.nan_to_num(1 / pred_depth, nan=0, posinf=0, neginf=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                gt_invdepth = np.nan_to_num(1 / gt_depth, nan=0, posinf=0, neginf=0)

            if mask.any():
                masked_gt_invdepth = (gt_invdepth[mask]).astype(np.float64)
                masked_pred_invdepth = (pred_invdepth[mask]).astype(np.float64)

                # system matrix: A = [[a_00, a_01], [a_10, a_11]]
                a_00 = np.sum(masked_pred_invdepth * masked_pred_invdepth)
                a_01 = np.sum(masked_pred_invdepth)
                a_11 = np.sum(mask.astype(np.float64))

                # right hand side: b = [b_0, b_1]
                b_0 = np.sum(masked_gt_invdepth * masked_pred_invdepth)
                b_1 = np.sum(masked_gt_invdepth)

                det = a_00 * a_11 - a_01 * a_01
                valid = det > 0

                if valid:
                    scale = ((a_11 * b_0 - a_01 * b_1) / det).astype(np.float32)
                    shift = ((-a_01 * b_0 + a_00 * b_1) / det).astype(np.float32)
                else:
                    scale = np.nan
                    shift = np.nan

            else:
                scale = np.nan
                shift = np.nan

            pred_invdepth = scale * pred_invdepth + shift
            with np.errstate(divide='ignore', invalid='ignore'):
                pred_depth = np.nan_to_num(1 / pred_invdepth, nan=0, posinf=0, neginf=0)
            del pred_invdepth
            del gt_invdepth
            pred['least_squares_scale'] = scale
            pred['least_squares_shift'] = shift

        if isinstance(self.clip_pred_depth, tuple):
            pred_depth = np.clip(pred_depth, self.clip_pred_depth[0], self.clip_pred_depth[1]) * pred_mask
        elif self.clip_pred_depth:
            pred_depth = np.clip(pred_depth, 0.1, 100) * pred_mask

        with np.errstate(divide='ignore', invalid='ignore'):
            pred_invdepth = np.nan_to_num(1 / pred_depth, nan=0, posinf=0, neginf=0)

        if 'depth_uncertainty' in pred:
            pred_depth_uncertainty = pred['depth_uncertainty']
            pred_depth_uncertainty = skimage.transform.resize(pred_depth_uncertainty, gt_depth.shape, order=0,
                                                              anti_aliasing=False)
            pred['depth_uncertainty'] = pred_depth_uncertainty

        pred['depth'] = pred_depth
        pred['invdepth'] = pred_invdepth

    def _run_model(self, sample_inputs):
        start_model_and_io = time.time()

        if hasattr(self.model, "input_adapter"):
            sample_inputs = self.model.input_adapter(**sample_inputs)

        start_model = time.time()
        pred = self.model(**sample_inputs)
        end_model = time.time()

        if hasattr(self.model, "output_adapter"):
            pred, _ = self.model.output_adapter(pred)

        end_model_and_io = time.time()

        time_and_mem_valid = self.cur_sample_num >= self.burn_in_samples
        runtime_model_in_sec = end_model - start_model if time_and_mem_valid else np.nan
        runtime_model_and_io_in_sec = end_model_and_io - start_model_and_io if time_and_mem_valid else np.nan
        runtimes = {'runtime_model_in_sec': runtime_model_in_sec,
                    'runtime_model_in_msec': 1000 * runtime_model_in_sec,
                    'runtime_model_and_io_in_sec': runtime_model_and_io_in_sec,
                    'runtime_model_and_io_in_msec': 1000 * runtime_model_and_io_in_sec,}

        gpu_mem_alloc = int(torch.cuda.max_memory_allocated() / 1024 / 1024) if time_and_mem_valid else np.nan
        gpu_mem_reserved = int(torch.cuda.max_memory_reserved() / 1024 / 1024) if time_and_mem_valid else np.nan
        gpu_mem = {'gpu_mem_alloc_in_mib': gpu_mem_alloc, 'gpu_mem_alloc_in_mib': gpu_mem_reserved}

        return pred, runtimes, gpu_mem

    def _compute_metrics(self, sample_inputs, sample_gt, pred):

        gt_depth = sample_gt['depth'][0, 0]
        pred_depth = pred['depth'][0, 0]
        eval_mask = pred_depth != 0 if self.sparse_pred else np.ones_like(pred_depth, dtype=bool)

        absrel = m_rel_ae(gt=gt_depth, pred=pred_depth, mask=eval_mask, output_scaling_factor=100.0)
        inliers103 = thresh_inliers(gt=gt_depth, pred=pred_depth, thresh=1.03, mask=eval_mask, output_scaling_factor=100.0)

        if self.alignment is None:
            inliers125 = thresh_inliers(gt=gt_depth, pred=pred_depth, thresh=1.25, mask=eval_mask, output_scaling_factor=100.0)


        metrics = {'absrel': absrel, 'inliers103': inliers103,}

        if self.alignment is None:
            metrics['inliers125'] = inliers125

        if self.alignment == "median":
            metrics['scaling_factor'] = pred['scaling_factor']

        if self.alignment == "least_squares_scale_shift":
            metrics['least_squares_scale'] = pred['least_squares_scale']
            metrics['least_squares_shift'] = pred['least_squares_shift']

        metrics['pred_depth_density'] = np.sum(eval_mask) / eval_mask.size * 100

        return metrics

    def _log_metrics(self, metrics, sum_source_views):

        for metric, val in metrics.items():
            self.results.loc[self.cur_sample_idx, (sum_source_views, metric)] = val

    def _compute_uncertainty_metrics(self, sample_inputs, sample_gt, pred):
        if self.verbose:
            print("\tComputing uncertainty metrics:")

        gt_depth = sample_gt['depth'][0, 0]

        pred_depth = pred['depth'][0, 0]
        pred_depth_uncertainty = pred['depth_uncertainty'][0, 0]
        pred_mask = pred_depth != 0 if self.sparse_pred else np.ones_like(pred_depth, dtype=bool)

        oracle_uncertainty = pointwise_rel_ae(gt=gt_depth, pred=pred_depth, mask=pred_mask)

        sparsification_oracle = sparsification(gt=gt_depth, pred=pred_depth, uncertainty=oracle_uncertainty,
                                               mask=pred_mask, show_pbar=self.verbose,
                                               pbar_desc="            Oracle sparsification")
        sparsification_pred = sparsification(gt=gt_depth, pred=pred_depth, uncertainty=pred_depth_uncertainty,
                                             mask=pred_mask, show_pbar=self.verbose,
                                             pbar_desc="            Prediction sparsification")
        sparsification_errors = sparsification_pred - sparsification_oracle
        ause = sparsification_errors.sum(skipna=False) / 100
        ause = ause if np.isfinite(ause) else np.nan

        self.sparsification_curves.loc[(self.cur_sample_idx, "oracle"), :] = sparsification_oracle
        self.sparsification_curves.loc[(self.cur_sample_idx, "pred"), :] = sparsification_pred
        self.sparsification_curves.loc[(self.cur_sample_idx, "error"), :] = sparsification_errors

        if self.verbose:
            print(f"\t\t\tAUSE={ause}.")

        return {'ause': ause}

    def _output_results(self):

        results_per_sample = self.results['best']
        results = results_per_sample.mean()

        num_source_view_results_per_sample = self.results.drop('best', axis=1, level=0)
        num_source_view_results = num_source_view_results_per_sample.mean()

        if self.verbose:
            print()
            print("Results:")
            print(results)

        if self.out_dir is not None:

            if self.verbose:
                print(f"Writing results to {self.out_dir}.")

            results_per_sample.to_pickle(osp.join(self.sample_results_dir, "results.pickle"))
            results_per_sample.to_csv(osp.join(self.sample_results_dir, "results.csv"))
            results.to_pickle(osp.join(self.quantitatives_dir, "results.pickle"))
            results.to_csv(osp.join(self.quantitatives_dir, "results.csv"))

            num_source_view_results_per_sample.to_csv(osp.join(self.sample_results_dir, "num_source_view_results.csv"))
            num_source_view_results_per_sample.to_pickle(osp.join(self.sample_results_dir, "num_source_view_results.pickle"))
            num_source_view_results.to_csv(osp.join(self.quantitatives_dir, "num_source_view_results.csv"))
            num_source_view_results.to_pickle(osp.join(self.quantitatives_dir, "num_source_view_results.pickle"))

            if self.eval_uncertainty:
                sample_sparsification_curves = self.sparsification_curves
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    sparsification_curves = sample_sparsification_curves.mean(axis=0, level=1)

                sparsification_curves.to_pickle(osp.join(self.quantitatives_dir, "sparsification_curves.pickle"))
                sparsification_curves.to_csv(osp.join(self.quantitatives_dir, "sparsification_curves.csv"))
                sample_sparsification_curves.to_pickle(osp.join(self.sample_results_dir, "sparsification_curves.pickle"))
                sample_sparsification_curves.to_csv(osp.join(self.sample_results_dir, "sparsification_curves.csv"))

            self._output_dataset_cfg()

    def _output_dataset_cfg(self):

        update_name = "_".join([s for s in [self.model.name, self.exp_name] if s is not None])
        dataset_updates_path = osp.join(self.qualitatives_dir, f"{update_name}.pickle")
        dataset_layout_path = osp.join(self.qualitatives_dir, "layout.pickle")
        dataset_cfg_path = osp.join(self.qualitatives_dir, "dataset.cfg")
        with open(dataset_updates_path, 'wb') as dataset_udpates_file:
            pickle.dump(self.dataset_updates, dataset_udpates_file)

        dataset_layout = self._get_layout()
        dataset_layout.write(dataset_layout_path)

        dataset_cls_name = get_full_class_name(self.dataset)
        self.dataset.write_config(path=dataset_cfg_path, dataset_cls_name=dataset_cls_name,
                                  updates=[dataset_updates_path], update_strict=True,
                                  layouts=[dataset_layout_path])

    def _get_layout(self):
        layout = Layout(name="eval_mvd")

        def load_key_img(sample_dict):
            from itypes.vizdata.image import ImageVisualizationData
            key_img = sample_dict['images'][sample_dict['keyview_idx']].transpose(1, 2, 0).astype(np.uint8)
            key_img = ImageVisualizationData(key_img)
            return {'data': key_img}

        key_img_visualization = Visualization(col=0, row=0, visualization_type="image", load_fct=load_key_img, name="Key Image")
        layout.visualizations.append(key_img_visualization)

        def load_gt_depth(sample_dict):
            from itypes.vizdata.float import FloatVisualizationData
            depth = sample_dict['depth'].transpose(1, 2, 0)
            depth = FloatVisualizationData(depth)
            return {'data': depth}

        gt_depth_visualization = Visualization(col=0, row=1, visualization_type="float", load_fct=load_gt_depth, name="GT Depth")
        layout.visualizations.append(gt_depth_visualization)

        def load_gt_invdepth(sample_dict):
            from itypes.vizdata.float import FloatVisualizationData
            invdepth = sample_dict['invdepth'].transpose(1, 2, 0)
            invdepth = FloatVisualizationData(invdepth)
            return {'data': invdepth}

        gt_invdepth_visualization = Visualization(col=1, row=1, visualization_type="float", load_fct=load_gt_invdepth, name="GT Inverse Depth")
        layout.visualizations.append(gt_invdepth_visualization)

        def load_gt_depth_mask(sample_dict):
            from itypes.vizdata.float import FloatVisualizationData
            mask = (sample_dict['depth'] > 0).astype(np.float32).transpose(1, 2, 0)
            mask = FloatVisualizationData(mask)
            return {'data': mask}

        gt_depth_mask_visualization = Visualization(col=2, row=1, visualization_type="float", load_fct=load_gt_depth_mask, name="GT Mask")
        layout.visualizations.append(gt_depth_mask_visualization)

        def load_pred_depth(sample_dict):
            from itypes.vizdata.float import FloatVisualizationData
            depth = sample_dict['pred_depth'].transpose(1, 2, 0)
            depth = FloatVisualizationData(depth)
            return {'data': depth}

        pred_depth_visualization = Visualization(col=0, row=2, visualization_type="float", load_fct=load_pred_depth, name="Predicted Depth")
        layout.visualizations.append(pred_depth_visualization)

        def load_pred_invdepth(sample_dict):
            from itypes.vizdata.float import FloatVisualizationData
            invdepth = sample_dict['pred_invdepth'].transpose(1, 2, 0)
            invdepth = FloatVisualizationData(invdepth)
            return {'data': invdepth}

        pred_invdepth_visualization = Visualization(col=1, row=2, visualization_type="float", load_fct=load_pred_invdepth, name="Predicted Inverse Depth")
        layout.visualizations.append(pred_invdepth_visualization)

        def load_pointwise_absrel(sample_dict):
            from itypes.vizdata.float import FloatVisualizationData
            pointwise_absrel = sample_dict['pointwise_absrel'].transpose(1, 2, 0)
            pointwise_absrel = FloatVisualizationData(pointwise_absrel)
            return {'data': pointwise_absrel}

        pointwise_absrel_visualization = Visualization(col=2, row=2, visualization_type="float", load_fct=load_pointwise_absrel, name="Absolute Relative Error")
        layout.visualizations.append(pointwise_absrel_visualization)

        if self.eval_uncertainty:
            def load_pred_depth_uncertainty(sample_dict):
                from itypes.vizdata.float import FloatVisualizationData
                uncertainty = sample_dict['pred_depth_uncertainty'].transpose(1, 2, 0)
                uncertainty = FloatVisualizationData(uncertainty)
                return {'data': uncertainty}

            pred_depth_uncertainty_visualization = Visualization(col=3, row=2, visualization_type="float", load_fct=load_pred_depth_uncertainty, name="Predicted Depth Uncertainty")
            layout.visualizations.append(pred_depth_uncertainty_visualization)

        return layout


def filter_views_in_sample(sample, indices_to_keep):
    sample = deepcopy(sample)
    keyview_idx = int(sample['keyview_idx'])
    assert keyview_idx in indices_to_keep, "Keyview must not be filtered out."
    keyview_idx = indices_to_keep.index(keyview_idx)

    if "images" in sample:
        sample['images'] = [select_by_index(sample['images'], i) for i in indices_to_keep]
    if "poses" in sample:
        sample['poses'] = [select_by_index(sample['poses'], i) for i in indices_to_keep]
    if "intrinsics" in sample:
        sample['intrinsics'] = [select_by_index(sample['intrinsics'], i) for i in indices_to_keep]
    sample['keyview_idx'] = np.array([keyview_idx])

    return sample


class MultiMultiViewDepthEvaluationUpdate(Update):
    def __init__(self):
        self.update_dict = {}

    def load(self, orig_sample_dict, root=None):
        out_dict = {}
        for key, val in self.update_dict.items():
            if isinstance(val, str):
                if osp.isfile(val):
                    val = np.load(val)
            out_dict[key] = val
        return out_dict

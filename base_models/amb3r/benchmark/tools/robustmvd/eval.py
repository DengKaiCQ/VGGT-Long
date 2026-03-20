#!/usr/bin/env python3

# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/eval.py

import argparse
import sys
import os.path as osp

import torch

from rmvd import create_model, list_models, create_dataset, list_datasets, create_evaluation, list_evaluations


@torch.no_grad()
def eval(args):

    if args.model is None:
        print(f"No model specified. Available models are: {', '.join(list_models())}")
        return

    if args.eval_type is None:
        print(f"No evaluation type specified. Available evaluation types are: {', '.join(list_evaluations())}")
        return

    if args.eval_type != "robustmvd":
        if args.dataset is None:  # or dataset not available
            datasets = list_datasets(dataset_type=args.eval_type, no_dataset_type=True)
            print(f"No dataset specified. Available datasets are: {', '.join(datasets)}")
            return

        print()
        print(f"Evaluating {args.model} model on dataset {args.dataset} in the {args.eval_type} evaluation setting.\n")
        dataset = create_dataset(dataset_name_or_path=args.dataset, dataset_type=args.eval_type,
                                 input_size=args.input_size)

    else:
        print()
        print(f"Evaluating {args.model} model on the Robust Multi-view Depth Benchmark.\n")
        dataset = None

    model = create_model(name=args.model, weights=args.weights, train=False, num_gpus=args.num_gpus)
    eval = create_evaluation(evaluation_type=args.eval_type,
                             out_dir=args.output,
                             inputs=args.inputs,
                             alignment=args.alignment,
                             view_ordering=args.view_ordering,
                             min_source_views=args.min_source_views,
                             max_source_views=args.max_source_views,
                             eval_uncertainty=args.eval_uncertainty)

    with open(osp.join(args.output, "cmd.txt"), 'a') as f:
        f.write("python " + " ".join(sys.argv) + "\n")

    samples = args.num_samples if args.num_samples is not None else args.samples
    qualitatives = args.qualitatives if args.qualitatives is not None else args.num_qualitatives

    eval(dataset=dataset, model=model, samples=samples, qualitatives=qualitatives,
         eth3d_size=args.eth3d_size, kitti_size=args.kitti_size, dtu_size=args.dtu_size, scannet_size=args.scannet_size,
         tanks_and_temples_size=args.tanks_and_temples_size, exp_name=args.exp_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help=f"Model for evaluation. Available models are: {', '.join(list_models())}")
    parser.add_argument('--weights', help="Path to weights of the model. Optional. If None, default weights are used.")
    parser.add_argument('--num_gpus', type=int, help="Number of GPUs. 0 means use CPU. Default: use 1 GPU.", default=1)
    parser.add_argument('--eval_type', help=f"Evaluation setting. Options are: {', '.join(list_evaluations())}")
    parser.add_argument('--inputs', nargs='*',
                        help=f"Model inputs. Images are always provided to the model. "
                             f"It is possible to specify multiple additional inputs, "
                             f"e.g. --inputs intrinsics poses. "
                             f"Options for additional model inputs are: intrinsics, poses, depth_range.",
                        type=str)
    parser.add_argument('--output', help="Path to folder for output data.")
    parser.add_argument('--exp_name', help="Experiment name. Optional.", type=str)

    parser.add_argument('--num_samples', type=int, help='Number of samples to be evaluated. Default: evaluate all.')
    parser.add_argument('--samples', type=int, nargs='*',
                        help='Index of sample that should be evaluated. Ignored if num_samples is used. '
                             'Default: evaluate all.')

    parser.add_argument('--max_source_views', type=int, help='Maximum number of source views to use for evaluation. '
                                                             'Default: use all available source views.')
    parser.add_argument('--min_source_views', type=int, default=1,
                        help='Minimum number of source views to use for evaluation. Default: 1.')
    parser.add_argument('--view_ordering', default="quasi-optimal",
                        help=f"Source view ordering. Options are: quasi-optimal (default), nearest.")
    parser.add_argument('--alignment',
                        help=f"Alignment between predicted and ground truth depths. "
                             f"Options are None, median, translation. Default: None")

    parser.add_argument('--num_qualitatives', type=int, default=10,
                        help='Number of qualitatives to be output. Negative values output all qualitatives. '
                             'Ignored if --qualitative is used. Default: 10.')
    parser.add_argument('--qualitatives', type=int, nargs='*',
                        help='Index of sample where qualitatives should be output.')

    parser.add_argument('--eval_uncertainty', action='store_true', help='Evaluate predicted depth uncertainty.')

    # arguments for the "mvd" evaluation:
    parser.add_argument('--dataset', help=f"Dataset. Available datasets are: {', '.join(list_datasets())}")
    parser.add_argument('--input_size', type=int, nargs=2, help="Input image size in the format (height, width). "
                                                                "If not provided, scales images up to the nearest size "
                                                                "that works with the model.")

    # arguments for the "robustmvd" evaluation:
    parser.add_argument('--eth3d_size', type=int, nargs=2, default=[1024, 1536],
                        help="Input image size on ETH3D in the format (height, width). "
                             "If not provided, scales images down to the size (1024, 1536).")
    parser.add_argument('--kitti_size', type=int, nargs=2,
                        help="Input image size on KITTI in the format (height, width). "
                             "If not provided, scales images up to the nearest size that works with the model.")
    parser.add_argument('--dtu_size', type=int, nargs=2,
                        help="Input image size on DTU in the format (height, width). "
                             "If not provided, scales images up to the nearest size that works with the model.")
    parser.add_argument('--scannet_size', type=int, nargs=2,
                        help="Input image size on ScanNet in the format (height, width). "
                             "If not provided, scales images up to the nearest size that works with the model.")
    parser.add_argument('--tanks_and_temples_size', type=int, nargs=2,
                        help="Input image size on Tanks and Temples in the format (height, width). "
                             "If not provided, scales images up to the nearest size that works with the model.")

    args = parser.parse_args()

    eval(args)

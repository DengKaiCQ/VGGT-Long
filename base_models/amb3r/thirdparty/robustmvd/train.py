#!/usr/bin/env python3

# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/train.py

import argparse
import sys
import os
import os.path as osp

from rmvd import create_model, list_models, create_dataset, create_compound_dataset, list_datasets, create_training, list_trainings, create_optimizer, list_optimizers, create_scheduler, list_schedulers, create_loss, list_losses, list_augmentations, list_batch_augmentations
from rmvd.utils import set_random_seed, writer


def train(args):
    
    set_random_seed(args.seed)

    if args.model is None:
        print(f"No model specified via --model. Available models are: {', '.join(list_models(trainable_only=True))}")
        return

    if args.training_type is None:
        print(f"No training type specified via --training_type. Available training types are: {', '.join(list_trainings())}")
        return
    
    if args.augmentations is not None and args.augmentations_per_dataset is not None:
        print("Error: --augmentations and --augmentations_per_dataset cannot be used together.")
        return
    
    if args.augmentations_per_dataset is not None and len(args.augmentations_per_dataset) != len(args.dataset):
        print("Error: There must be as many --augmentations_per_dataset arguments as --dataset arguments.")
        return

    if args.dataset is None:  # or dataset not available
        datasets = list_datasets()
        print(f"No dataset specified via --dataset. Available datasets are: {', '.join(datasets)}")
        return

    if args.optimizer is None:
        print(f"No optimizer specified via --optimizer. Available optimizers are: {', '.join(list_optimizers())}")
        return

    if args.scheduler is None:
        print(f"No scheduler specified. Available schedulers are: {', '.join(list_schedulers())}")
        return
    
    if args.loss is None:
        print(f"No loss specified. Available losses are: {', '.join(list_losses())}")
        return
    
    out_dir = args.output
    tensorboard_logs_dir = osp.join(out_dir, "tensorboard_logs")
    wandb_logs_dir = osp.join(out_dir, "wandb_logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tensorboard_logs_dir, exist_ok=True)
    os.makedirs(wandb_logs_dir, exist_ok=True)
    writer.setup_writers(log_tensorboard=not args.no_tensorboard, 
                         log_wandb=args.wandb, 
                         max_iterations=args.max_iterations,
                         tensorboard_logs_dir=tensorboard_logs_dir, 
                         wandb_logs_dir=wandb_logs_dir,
                         exp_id=args.exp_id,
                         comment=args.comment,)  # TODO: config=CONFIG

    print()
    print(f"Training {args.model} model on the dataset {'+'.join(args.dataset)} in the {args.training_type} training setting.\n")
    
    datasets = []
    for dataset_idx, dataset_name in enumerate(args.dataset):
        augmentation = args.augmentations_per_dataset[dataset_idx] if args.augmentations_per_dataset is not None else args.augmentations
        dataset = create_dataset(dataset_name_or_path=dataset_name, input_size=args.input_size, target_size=args.target_size, augmentations=augmentation)
        datasets.append(dataset)
    dataset = datasets[0] if len(datasets) == 1 else create_compound_dataset(datasets=datasets)

    model = create_model(name=args.model, pretrained=False, weights=args.weights, train=True, num_gpus=args.num_gpus)

    optimizer = create_optimizer(name=args.optimizer, model=model, lr=args.lr)
    scheduler = create_scheduler(name=args.scheduler, optimizer=optimizer)

    loss = create_loss(name=args.loss, model=model)

    training = create_training(training_type=args.training_type,
                               out_dir=out_dir,
                               model=model,
                               dataset=dataset,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               loss=loss,
                               batch_size=args.batch_size,
                               max_iterations=args.max_iterations,
                               inputs=args.inputs,
                               batch_augmentations=args.batch_augmentations,
                               grad_clip_max_norm=args.grad_clip_max_norm,
                               num_workers=args.num_workers,
                               log_interval=args.log_interval,
                               log_full_batch=args.log_full_batch,
                               verbose=True,)

    with open(osp.join(args.output, "cmd.txt"), 'a') as f:
        f.write("python " + " ".join(sys.argv) + "\n")

    training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help="Path to folder for output data.", required=True)
    parser.add_argument('--num_gpus', type=int, help="Number of GPUs. 0 means use CPU. Default: use 1 GPU.", default=1)
    parser.add_argument('--seed', type=int, help="Random seed. Default: 42.", default=42)
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size.")
    parser.add_argument('--max_iterations', type=int, required=True, help="Maximum number of iterations to train.")
    parser.add_argument('--num_workers', type=int, help="Number of workers for data loading. Default: 8.", default=8)
    parser.add_argument('--training_type', help=f"Training setting. Options are: {', '.join(list_trainings())}", required=True)

    parser.add_argument('--model', help=f"Model for evaluation. Available models "
                                        f"are: {', '.join(list_models(trainable_only=True))}", required=True)
    parser.add_argument('--weights', help="Path to weights of the model. Optional. "
                                          "If None, default weight initialization of the model is used.")

    parser.add_argument('--dataset', help=f"Dataset. Available datasets are: {', '.join(list_datasets())}", required=True, action='append')
    parser.add_argument('--input_size', type=int, nargs=2, help="Input data size in the format (height, width). If not provided, uses the original size.")
    parser.add_argument('--target_size', type=int, nargs=2, help="Target data size in the format (height, width). If not provided, uses the original size.")
    parser.add_argument('--augmentations', nargs='*',
                        help=f"Data augmentations. Options are: {', '.join(list_augmentations())}")
    parser.add_argument('--batch_augmentations', nargs='*',
                        help=f"Data augmentations that are applied to the whole batch. Options are: {', '.join(list_batch_augmentations())}")
    parser.add_argument('--augmentations_per_dataset', nargs='*', action='append',
                        help=f"Data augmentations per dataset in case training is done on multiple datasets and the "
                        f"datasets should have different augmentations. Options are: {', '.join(list_augmentations())}")

    parser.add_argument('--optimizer', help=f"Optimizer. Options are: {', '.join(list_optimizers())}", required=True)
    parser.add_argument('--lr', type=float, help="Learning rate. Default: 1e-4", default=1e-4)
    parser.add_argument('--grad_clip_max_norm', type=float, help="Maximum norm of gradient. Default: do not clip gradient.")

    parser.add_argument('--scheduler', help=f"Scheduler. Options are: {', '.join(list_schedulers())}", required=True)

    parser.add_argument('--loss', help=f"Loss. Options are: {', '.join(list_losses())}", required=True)

    parser.add_argument('--inputs', nargs='*',
                        help=f"Model inputs. Images are always provided to the model. "
                             f"It is possible to specify multiple additional inputs, "
                             f"e.g. --inputs intrinsics poses. "
                             f"Options for additional model inputs are: intrinsics, poses, depth_range.",
                        type=str)

    parser.add_argument('--log_interval', type=int, help="Log interval in iterations. Default: 5000.", default=5000)
    parser.add_argument('--log_full_batch', action='store_true', help='Write all samples in batch to log. '
                                                                      'Default: only log first sample.')
    parser.add_argument('--no_tensorboard', action='store_true', help='Do not log to tensorboard. Default: do log.')
    parser.add_argument('--wandb', action='store_true', help='Log to weights and biases. Default: Do not log.')
    parser.add_argument('--exp_id', type=str, help="Experiment ID. Used for wandb logging.")
    parser.add_argument('--comment', type=str, help="Comment for the experiment. Used for wandb logging.")

    args = parser.parse_args()

    train(args)

# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/eval_mvdepth.py

import os
import sys
import torch
import argparse

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

robustmvd_path = os.path.join(current_dir, "tools", "robustmvd")
if robustmvd_path not in sys.path:
    sys.path.insert(0, robustmvd_path)

from amb3r.model import AMB3R
from rmvd import create_evaluation, prepare_custom_model


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, default="./outputs/mvdepth")
    parser.add_argument('--metric', action='store_true')
    return parser


@torch.no_grad()
def main(args):
    model = AMB3R(metric_scale=args.metric)
    model.load_weights(os.path.join(project_root, 'checkpoints', 'amb3r.pt'))
    model = prepare_custom_model(model, num_gpus=1)

    
    metric_scale = args.metric
    results_path = args.results_path + ("_metric_scale" if metric_scale else "")


    evaluation = create_evaluation(
            evaluation_type="robustmvd",
            out_dir=results_path,
            eval_uncertainty=False,
            alignment=None if metric_scale else 'median',
        )

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        results = evaluation(
                model=model,
                eth3d_size=(392, 518),
                kitti_size=(392, 518),
                dtu_size=(392, 518),
                scannet_size=(392, 518),
                tanks_and_temples_size=(392, 518))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)



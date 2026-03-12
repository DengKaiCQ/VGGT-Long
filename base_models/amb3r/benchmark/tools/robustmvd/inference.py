#!/usr/bin/env python3

# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/inference.py

import argparse
import os
import os.path as osp
import math
import sys

import torch
import numpy as np
from PIL import Image
from skimage.transform import resize

from rmvd import create_model, list_models
from rmvd.utils import invert_transform, vis_2d_array


def load_data(path):
    key_path = osp.join(path, "key")
    src_path = osp.join(path, "source")
    src_paths = sorted([osp.join(src_path, x) for x in os.listdir(src_path)])

    image_key = np.array(Image.open(osp.join(key_path, 'image.png')), dtype=np.float32).transpose(2, 0, 1)
    intrinsics_key = np.load(osp.join(key_path, 'K.npy'))
    key_to_ref_transform = np.load(osp.join(key_path, 'to_ref_transform.npy'))
    ref_to_key_transform = invert_transform(key_to_ref_transform)
    key_to_key_transform = key_to_ref_transform.dot(ref_to_key_transform)
    h_orig, w_orig = image_key.shape[-2:]

    images_source = []
    source_to_key_transforms = []
    intrinsics_source = []
    for src_path in src_paths:
        image_source = np.array(Image.open(osp.join(src_path, 'image.png')), dtype=np.float32).transpose(2, 0, 1)
        intrinsic_source = np.load(osp.join(src_path, 'K.npy'))
        source_to_ref_transform = np.load(osp.join(src_path, 'to_ref_transform.npy'))
        source_to_key_transform = np.dot(source_to_ref_transform, ref_to_key_transform)

        images_source.append(image_source)
        source_to_key_transforms.append(source_to_key_transform)
        intrinsics_source.append(intrinsic_source)

    sample = {
        'images': [image_key] + images_source,
        'intrinsics': [intrinsics_key] + intrinsics_source,
        'poses': [key_to_key_transform] + source_to_key_transforms,
        'keyview_idx': 0,
    }

    return sample, h_orig, w_orig


def write_pred(pred, output_path, h_orig, w_orig):
    pred_depth = pred['depth']
    pred_depth = resize(pred_depth, list(pred_depth.shape[:-2]) + [h_orig, w_orig], order=0, anti_aliasing=False)

    with np.errstate(divide='ignore', invalid='ignore'):
        pred_invdepth = 1. / pred_depth

    np.save(osp.join(output_path, "pred_depth.npy"), pred_depth)
    np.save(osp.join(output_path, "pred_invdepth.npy"), pred_invdepth)
    vis_2d_array(pred_invdepth, mark_invalid=True).save(osp.join(args.output_path, "pred_invdepth.png"))
    vis_2d_array(pred_depth, mark_invalid=True).save(osp.join(args.output_path, "pred_depth.png"))

    if 'depth_uncertainty' in pred:
        pred_depth_uncertainty = pred['depth_uncertainty']
        pred_depth_uncertainty = resize(pred_depth_uncertainty,
                                        list(pred_depth_uncertainty.shape[:-2]) + [h_orig, w_orig],
                                        order=0, anti_aliasing=False)
        np.save(osp.join(args.output_path, "pred_depth_uncertainty.npy"), pred_depth_uncertainty)
        vis_2d_array(pred_depth_uncertainty, image_range_text_off=True).save(
            osp.join(args.output_path, "pred_depth_uncertainty.png"))
    else:
        if osp.exists(osp.join(args.output_path, "pred_depth_uncertainty.png")):
            os.remove(osp.join(args.output_path, "pred_depth_uncertainty.png"))
        if osp.exists(osp.join(args.output_path, "pred_depth_uncertainty.npy")):
            os.remove(osp.join(args.output_path, "pred_depth_uncertainty.npy"))


@torch.no_grad()
def run(args):

    if args.model is None:
        print(f"No model specified. Available models are: {', '.join(list_models())}")
        return

    print(f"Running inference on data from {args.input_path} with model {args.model}.")

    os.makedirs(args.output_path, exist_ok=True)
    with open(osp.join(args.output_path, "cmd.txt"), 'w') as f:
        f.write("python " + " ".join(sys.argv))

    model = create_model(name=args.model, weights=args.weights, train=False, num_gpus=args.num_gpus)
    sample, h_orig, w_orig = load_data(args.input_path)
    pred, _ = model.run(**sample)
    write_pred(pred, args.output_path, h_orig, w_orig)

    print("Done. Output written to {}.".format(args.output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default="sample_data", help="Path to folder with input data.")
    parser.add_argument('--output_path', default="sample_data/out", help="Path to folder for output data.")

    parser.add_argument('--model', help=f"Model for evaluation. Available models are: {', '.join(list_models())}")
    parser.add_argument('--weights', help="Path to weights of the model. Optional. If None, default weights are used.")
    parser.add_argument('--num_gpus', type=int, help="Number of GPUs. 0 means use CPU. Default: use 1 GPU.", default=1)

    # TODO:
    # parser.add_argument('--input_width', type=int, help="Input image width.")
    # parser.add_argument('--input_height', type=int, help="Input image height.")
    args = parser.parse_args()

    run(args)

#!/usr/bin/env python3

# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/data_viewer.py

import argparse
from rmvd import list_datasets, create_dataset, run_viewer, list_augmentations


def data_viewer(args):
    dataset = create_dataset(args.data, augmentations=args.augmentations)

    if args.layout is None:
        layout_names = dataset.get_layout_names()
        if any([layout.startswith("eval") for layout in layout_names]):
            layout = [layout for layout in layout_names if layout.startswith("eval")][0]
        else:
            layout = "default"
    else:
        layout = args.layout

    run_viewer(dataset, layout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help=f"Data to be viewed. Can either be a path to evaluation outputs, or a dataset"
                                     f"name. Available dataset names are: {', '.join(list_datasets())}")
    parser.add_argument('--layout', help=f"Data viewer layout. If not specified, the default layout is used.")
    parser.add_argument('--augmentations', nargs='*',
                        help=f"Data augmentations. Options are: {', '.join(list_augmentations())}")
    args = parser.parse_args()
    data_viewer(args)

# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/data/layouts.py

import math

import numpy as np

from .layout import Layout, Visualization
from rmvd.utils import vis


class MVDSequentialDefaultLayout(Layout):
    def __init__(self, name, num_views, keyview_idx):
        self.num_views = num_views
        self.keyview_idx = keyview_idx
        super().__init__(name=name)

        max_fwd = min(2, self.num_views - self.keyview_idx - 1)
        max_bwd = min(2, self.keyview_idx)

        def load_key_img(sample_dict):
            from itypes.vizdata.image import ImageVisualizationData
            key_img = sample_dict['images'][sample_dict['keyview_idx']]
            key_img = vis(key_img, out_format={'type': 'np'}, image_range_text_off=True)
            key_img = key_img.transpose(1, 2, 0).astype(np.uint8)
            key_img = ImageVisualizationData(key_img)
            return {'data': key_img}

        key_img_visualization = Visualization(col=2, row=0, visualization_type="image", load_fct=load_key_img, name="Key Image")
        self.visualizations.append(key_img_visualization)

        def load_gt_depth(sample_dict):
            from itypes.vizdata.float import FloatVisualizationData
            depth = sample_dict['depth'].transpose(1, 2, 0)
            depth = FloatVisualizationData(depth)
            return {'data': depth}

        gt_depth_visualization = Visualization(col=2, row=1, visualization_type="float", load_fct=load_gt_depth, name="GT Depth")
        self.visualizations.append(gt_depth_visualization)

        def load_gt_invdepth(sample_dict):
            from itypes.vizdata.float import FloatVisualizationData
            invdepth = sample_dict['invdepth'].transpose(1, 2, 0)
            invdepth = FloatVisualizationData(invdepth)
            return {'data': invdepth}

        gt_invdepth_visualization = Visualization(col=3, row=1, visualization_type="float", load_fct=load_gt_invdepth, name="GT Inverse Depth")
        self.visualizations.append(gt_invdepth_visualization)

        def load_gt_depth_mask(sample_dict):
            from itypes.vizdata.float import FloatVisualizationData
            mask = (sample_dict['depth'] > 0).astype(np.float32).transpose(1, 2, 0)
            mask = FloatVisualizationData(mask)
            return {'data': mask}

        gt_depth_mask_visualization = Visualization(col=4, row=1, visualization_type="mask", load_fct=load_gt_depth_mask, name="GT Mask")
        self.visualizations.append(gt_depth_mask_visualization)

        for i in list(range(-max_bwd, 0)) + list(range(1, 1+max_fwd)):
            def load_source_img(sample_dict, idx=i):
                from itypes.vizdata.image import ImageVisualizationData
                source_img = sample_dict['images'][sample_dict['keyview_idx'] + idx]
                source_img = vis(source_img, out_format={'type': 'np'}, image_range_text_off=True)
                source_img = source_img.transpose(1, 2, 0).astype(np.uint8)
                source_img = ImageVisualizationData(source_img)
                return {'data': source_img}

            source_img_visualization = Visualization(col=2+i, row=0, visualization_type="image", load_fct=load_source_img,
                                                     name="Source Image @{}{}".format(('+' if i > 0 else ''), i))
            self.visualizations.append(source_img_visualization)


class MVDUnstructuredDefaultLayout(Layout):
    def __init__(self, name, num_views, max_views):
        self.num_views = num_views
        self.max_views = max_views
        self.keyview_idx = 0
        super().__init__(name=name)

        per_row = 5
        num_views = min(self.num_views, self.max_views)
        image_rows = math.ceil(num_views/(per_row-1))

        def load_key_img(sample_dict):
            from itypes.vizdata.image import ImageVisualizationData
            key_img = sample_dict['images'][sample_dict['keyview_idx']]
            key_img = vis(key_img, out_format={'type': 'np'}, image_range_text_off=True)
            key_img = key_img.transpose(1, 2, 0).astype(np.uint8)
            key_img = ImageVisualizationData(key_img)
            return {'data': key_img}

        key_img_visualization = Visualization(col=0, row=0, visualization_type="image", load_fct=load_key_img, name="Key Image")
        self.visualizations.append(key_img_visualization)

        def load_gt_depth(sample_dict):
            from itypes.vizdata.float import FloatVisualizationData
            depth = sample_dict['depth'].transpose(1, 2, 0)
            depth = FloatVisualizationData(depth)
            return {'data': depth}

        gt_depth_visualization = Visualization(col=0, row=image_rows, visualization_type="float", load_fct=load_gt_depth, name="GT Depth")
        self.visualizations.append(gt_depth_visualization)

        def load_gt_invdepth(sample_dict):
            from itypes.vizdata.float import FloatVisualizationData
            invdepth = sample_dict['invdepth'].transpose(1, 2, 0)
            invdepth = FloatVisualizationData(invdepth)
            return {'data': invdepth}

        gt_invdepth_visualization = Visualization(col=1, row=image_rows, visualization_type="float", load_fct=load_gt_invdepth, name="GT Inverse Depth")
        self.visualizations.append(gt_invdepth_visualization)

        def load_gt_depth_mask(sample_dict):
            from itypes.vizdata.float import FloatVisualizationData
            mask = (sample_dict['depth'] > 0).astype(np.float32).transpose(1, 2, 0)
            mask = FloatVisualizationData(mask)
            return {'data': mask}

        gt_depth_mask_visualization = Visualization(col=2, row=image_rows, visualization_type="mask", load_fct=load_gt_depth_mask, name="GT Mask")
        self.visualizations.append(gt_depth_mask_visualization)

        for i in range(1, num_views):
            def load_source_img(sample_dict, idx=i):
                from itypes.vizdata.image import ImageVisualizationData
                key_idx = sample_dict['keyview_idx']
                assert key_idx == 0, "This layout only works with samples where keyview_idx=0"
                source_img = sample_dict['images'][key_idx+idx]
                source_img = vis(source_img, out_format={'type': 'np'}, image_range_text_off=True)
                source_img = source_img.transpose(1, 2, 0).astype(np.uint8)
                source_img = ImageVisualizationData(source_img)
                return {'data': source_img}

            col = (i-1) % (per_row-1)+1
            row = (i-1) // (per_row-1)
            source_img_visualization = Visualization(col=col, row=row, visualization_type="image",
                                                     load_fct=load_source_img,
                                                     name=f"Source Image #{i}")
            self.visualizations.append(source_img_visualization)


class AllImagesLayout(Layout):
    def __init__(self, name, num_views):
        self.num_views = num_views
        super().__init__(name=name)

        per_row = 5

        def load_key_img(sample_dict):
            from itypes.vizdata.image import ImageVisualizationData
            key_img = sample_dict['images'][sample_dict['keyview_idx']]
            key_img = vis(key_img, out_format={'type': 'np'}, image_range_text_off=True)
            key_img = key_img.transpose(1, 2, 0).astype(np.uint8)
            key_img = ImageVisualizationData(key_img)
            return {'data': key_img}

        key_img_visualization = Visualization(col=0, row=0, visualization_type="image", load_fct=load_key_img, name="Key Image")
        self.visualizations.append(key_img_visualization)

        for i in range(self.num_views-1):
            def load_source_img(sample_dict, idx=i):
                from itypes.vizdata.image import ImageVisualizationData
                key_idx = sample_dict['keyview_idx']
                src_idx = idx if idx < key_idx else idx + 1
                source_img = sample_dict['images'][src_idx]
                source_img = vis(source_img, out_format={'type': 'np'}, image_range_text_off=True)
                source_img = source_img.transpose(1, 2, 0).astype(np.uint8)
                source_img = ImageVisualizationData(source_img)
                return {'data': source_img}

            col = (i+1)%(per_row-1)
            row = (i+1)//(per_row-1)
            source_img_visualization = Visualization(col=col, row=row,
                                                     visualization_type="image", load_fct=load_source_img,
                                                     name=f"Source Image #{i+1}")
            self.visualizations.append(source_img_visualization)

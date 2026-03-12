# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/utils/vis.py

import os
import functools

import numpy as np
import torch
from torch.utils.tensorboard.summary import make_np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import skimage.transform

from .turbo_colormap import cmap as turbo_cmap


_DEFAULT_FONT_SIZE = 10
_DEFAULT_FONT_PATH = '/tmp/OpenSans-Regular.ttf'
_DEFAULT_FONTS = {_DEFAULT_FONT_SIZE: ImageFont.truetype(_DEFAULT_FONT_PATH, _DEFAULT_FONT_SIZE) if os.path.isfile(_DEFAULT_FONT_PATH) else None}
_DEFAULT_BBOX_COLOR = (238, 232, 213)
_DEFAULT_BBOX_STROKE = None
_DEFAULT_TEXT_COLOR = (0, 43, 54)
_DEFAULT_CMAP = 'turbo'


def _get_default_font(size=None):
    if size is None:
        if _DEFAULT_FONT_SIZE not in _DEFAULT_FONTS:
            _DEFAULT_FONTS[_DEFAULT_FONT_SIZE] = ImageFont.truetype(_DEFAULT_FONT_PATH, _DEFAULT_FONT_SIZE) if os.path.isfile(_DEFAULT_FONT_PATH) else None
        return _DEFAULT_FONTS[_DEFAULT_FONT_SIZE]
    else:
        return ImageFont.truetype(_DEFAULT_FONT_PATH, size) if os.path.isfile(_DEFAULT_FONT_PATH) else None


def _get_cmap(cmap_name):
    if cmap_name == 'turbo':
        cmap = turbo_cmap
    else:
        cmap = plt.get_cmap(cmap_name)
    return cmap


def _cmap_min_str(cmap_name):
    if cmap_name == 'plasma':
        return 'blue'
    elif cmap_name == 'jet':
        return 'blue'
    elif cmap_name == 'turbo':
        return 'purple'
    elif cmap_name == 'gray':
        return 'black'
    elif cmap_name == 'autumn':
        return 'red'
    elif cmap_name == 'cool':
        return 'blue'
    else:
        ''


def _cmap_max_str(cmap_name):
    if cmap_name == 'plasma':
        return 'yellow'
    elif cmap_name == 'jet':
        return 'red'
    elif cmap_name == 'turbo':
        return 'red'
    elif cmap_name == 'gray':
        return 'white'
    elif cmap_name == 'autumn':
        return 'yellow'
    elif cmap_name == 'cool':
        return 'pink'
    else:
        ''


def _get_draw_text(text, label, text_off, image_range_text, image_range_text_off):
    draw_text = ""
    
    if label is not None:
        draw_text += str(label)
        if (not text_off) or (not image_range_text_off):
            draw_text += "\n"
    
    if text is not None and not text_off:
        draw_text += text
        if not image_range_text_off:
            draw_text += "\n"

    if not image_range_text_off:
        draw_text += image_range_text

    return draw_text


def _to_img(arr, mode):

    if mode == 'BGR' and arr.ndim == 3:  # convert('BGR') somehow does not work..
        arr = arr[:, :, ::-1]
        mode = 'RGB'

    img = Image.fromarray(arr).convert(mode)

    return img


def _convert_to_out_format(img, out_format):
    if out_format['type'] == 'PIL':
        out = img
    elif out_format['type'] == 'np':
        out = np.array(img, dtype=out_format['dtype'] if 'dtype' in out_format else None).transpose(2, 0, 1)
    return out


def _apply_out_action(out, out_action, out_format):

    if out_action is None:
        return

    elif isinstance(out_action, dict):
        if out_action['type'] == 'save':
            if out_format['type'] == 'PIL':
                out.save(out_action['path'])
            elif out_format['type'] == 'np':
                np.save(out_action['path'], out)

    elif isinstance(out_action, str):
        if out_action == 'show':
            out.show()
            
            
def _equalize_sizes(imgs):
    if isinstance(imgs[0], Image.Image):
        max_width = max([img.width for img in imgs])
        max_height = max([img.height for img in imgs])
        for i, img in enumerate(imgs):
            if img.width != max_width or img.height != max_height:
                imgs[i] = img.resize(size=(max_width, max_height), resample=Image.NEAREST)
    else:  # np, shape CHW
        max_width = max([img.shape[2] for img in imgs])
        max_height = max([img.shape[1] for img in imgs])
        for i, img in enumerate(imgs):
            if img.shape[2] != max_width or img.shape[1] != max_height:
                imgs[i] = skimage.transform.resize(img, [img.shape[0], max_height, max_width], order=0, preserve_range=True)
    return imgs


def cat_images_colwise(imgs):
    imgs = _equalize_sizes(imgs)
    if isinstance(imgs[0], Image.Image):  # PIL
        img = np.concatenate([np.array(img) for img in imgs], axis=1)
        img = Image.fromarray(img)
    else:  # np, shape CHW
        img = np.concatenate(imgs, axis=2)
    return img


def cat_images_rowwise(imgs):
    imgs = _equalize_sizes(imgs)
    if isinstance(imgs[0], Image.Image):  # PIL
        img = np.concatenate([np.array(img) for img in imgs], axis=0)
        img = Image.fromarray(img)
    else:  # np, shape CHW
        img = np.concatenate(imgs, axis=1)
    return img


def vis(arr, **kwargs):
    """Creates a visualization of a 2d array or 3d image and returns it as a PIL image.

    Input array can be a numpy array or a torch tensor. Input array can have a batch dimension.

    Args:
        arr: Input array. Can be a numpy array or a torch tensor. Can have the following dimension:
            2 dimensions: 2d array.
            3 dimensions with 3 channels in the first dimension: image.
            3 dimensions with N != 3 channels in the first dimension: batch of N 2d arrays.
            4 dimensions with N channels in the first and 3 channels in the second dimension: batch of N images.
            4 dimensions with N channels in the first and 1 channel in the second dimension: batch of N 2d arrays.
        kwargs: See vis_2d_array and vis_image functions.
    """

    ndim = arr.ndim
    shape = arr.shape

    if ndim == 2:
        return vis_2d_array(arr, **kwargs)
    elif ndim == 3:
        if shape[0] == 3:
            return vis_image(arr, **kwargs)
        else:
            return vis_2d_array(arr, **kwargs)
    elif ndim == 4:
        if shape[1] == 3:
            return vis_image(arr, **kwargs)
        else:
            assert shape[1] == 1, f"Can not visualize an array of shape {shape}."
            return vis_2d_array(arr, **kwargs)
    else:
        raise ValueError(f"Can not visualize an array of shape {shape}.")


def vis_2d_array(arr, full_batch=False, batch_labels=None, **kwargs):
    """
    Creates a visualization of a 2d numpy array or torch tensor.

    Args:
        arr: 2D numpy array or torch tensor.
        full_batch: Indicates whether all samples in the batch should be visualized.
            False: visualize only first sample in the batch.
            True/"cols": visualize all samples in the batch by concatenating col-wise (side-by-side).
            "rows": visualize all samples in the batch by concatenating row-wise.
        kwargs: See _vis_single_2d_array function.
    """

    assert 2 <= arr.ndim <= 4, f"2d array must have 2, 3 or 4 dimensions, but got shape {arr.shape}"
    if arr.ndim == 4:
        assert arr.shape[1] == 1, f"First dimension in a 2d array with shape {arr.shape} must " \
                                  f"be 1, but got {arr.shape[1]}."
        arr = arr[:, 0, :, :]

    arr = make_np(arr)

    if full_batch:
        arr = arr[None, ...] if arr.ndim == 2 else arr
        imgs = []
        for idx, ele in enumerate(arr):
            if batch_labels is not None:
                assert "label" not in kwargs, "It is not possible to use batch_labels and label argument at the same time."
                img = _vis_single_2d_array(ele, label=batch_labels[idx], **kwargs)
            else:
                img = _vis_single_2d_array(ele, **kwargs)
            imgs.append(img)

        if full_batch == "rows":
            return cat_images_rowwise(imgs)
        else:
            return cat_images_colwise(imgs)

    else:
        arr = arr[0] if arr.ndim == 3 else arr
        return _vis_single_2d_array(arr, **kwargs)


def _vis_single_2d_array(arr, colorize=True,
                         clipping=False, upper_clipping_thresh=None, lower_clipping_thresh=None,
                         mark_clipping=False, clipping_color=None,
                         invalid_values=None, mark_invalid=False, invalid_color=None,
                         text=None, label=None, cmap=_DEFAULT_CMAP,
                         image_range_text_off=False, image_range_colors_off=False, text_off=False,
                         out_format=None, out_action=None):
    """
    Creates a visualization of a 2d numpy array or torch tensor.

    Args:
        arr: 2D numpy array or torch tensor.
        colorize: If set to true, the values will be visualized by a colormap, otherwise as gray-values.
        clipping: If true, values above a certain threshold will be clipped before the visualization.
        upper_clipping_thresh: Threshold that is used for clipping the values. If set to False,
        the value mean + 2*std_deviation of the array will be used as threshold. The thresholds are also used
        as limits of the color range.
        lower_clipping_thresh: Threshold that is used for clipping the values. If set to False,
        the value mean - 2*std_deviation of the array will be used as threshold. The thresholds are also used
        as limits of the color range.
        mark_clipping: Mark clipped values with specific colors in the visualization.
        clipping_color: Color for marking clipped values.
        invalid_values: list of values that are invalid (e.g. [0]). If no such values exist, just pass None.
        mark_invalid: Mark invalid (NaN/Inf and all values in the invalid_values list) with
        specific colors in the visualization.
        invalid_color: Color for marking invalid values.
        text: Additional text that is printed on the visualization.
        cmap: Colormap to use for the visualization.
        desc=description string, colors=dict with keys marker_color, text_color, bbox_color, bbox_stroke, score).
        Everything except for coordinates can be None.
        image_range_text_off: If True, no text information about the range of the image values is added.
        text_off: If True, the provided text is not added to the image.
        out_format: Dict that describes the format of the output. All such dicts must have 'type' and 'mode' key.
        Currently supported are:
        {'type': 'PIL', 'mode': 'RGB' (see PIL docs for supported modes)} (this is the default format),
        {'type': 'np', 'mode': 'RGB' (see PIL docs for supported modes), 'dtype': 'uint8'}.
        out_action: Dict that describes an action on the output. All such dicts must have 'type' key.
        Note that some actions require a specific out_format.
        Currently supported are:
        None,
        {'type': 'show'}.
    """
    assert arr.ndim == 2, f"Single 2d array must have 2 dimension, but got shape {arr.shape}"
    arr = make_np(arr)

    arr = arr.astype(np.float32, copy=True)
    cmap_name = _DEFAULT_CMAP if cmap is None else cmap
    out_format = {'type': 'PIL', 'mode': 'RGB'} if out_format is None else out_format
    out_format['mode'] = 'RGB' if 'mode' not in out_format else out_format['mode']

    # Filter out all values that are somehow invalid and set them to 0:
    arr, invalid_mask, invalid_values_mask, clipping_mask, upper_clipping_mask, lower_clipping_mask,\
        upper_clipping_thresh, lower_clipping_thresh = \
        invalidate_np_array(arr, clipping, upper_clipping_thresh, lower_clipping_thresh, invalid_values)

    # Now work only with valid values of the array and make them visualizable (range 0, 256):
    arr_valid_only = np.ma.masked_array(arr, invalid_mask)

    if not clipping:
        min_value = arr_min = float(np.ma.min(arr_valid_only))
        max_value = arr_max = float(np.ma.max(arr_valid_only))
    else:
        min_value = float(lower_clipping_thresh)
        max_value = float(upper_clipping_thresh)
        arr_min = float(np.ma.min(arr_valid_only))
        arr_max = float(np.ma.max(arr_valid_only))

    min_max_diff = max_value - min_value
    is_constant = (max_value == min_value)

    if is_constant:
        if min_value == 0:  # array is constant 0
            arr_valid_only *= 0
        else:
            arr_valid_only /= min_value
            arr_valid_only *= 255.0
    else:
        arr_valid_only -= min_value
        arr_valid_only /= min_max_diff
        arr_valid_only *= 255.0

    arr = arr.astype(np.uint8)

    # Now make some (r,g,b) values out of the (0, 255) values:
    if colorize:
        cmap = _get_cmap(cmap_name)
        arr = np.uint8(cmap(arr) * 255)[:, :, 0:3]

        if mark_invalid:
            invalid_color = np.array([0, 0, 0]) if invalid_color is None else invalid_color
            arr[invalid_values_mask] = invalid_color

        if clipping:
            if mark_clipping:
                clipping_color = np.array([255, 255, 255]) if clipping_color is None else clipping_color
                arr[clipping_mask] = clipping_color
            else:
                min_color = np.uint8(cmap([0.0]) * 255)[:, 0:3]
                max_color = np.uint8(cmap([1.0]) * 255)[:, 0:3]
                arr[upper_clipping_mask] = max_color
                arr[lower_clipping_mask] = min_color

    else:
        arr = np.stack([arr, arr, arr], axis=-1)

        if mark_invalid:
            invalid_color = np.array([2, 10, 30]) if invalid_color is None else invalid_color
            arr[invalid_values_mask] = invalid_color

        if clipping:
            if mark_clipping:
                clipping_color = np.array([67, 50, 54]) if clipping_color is None else clipping_color
                arr[clipping_mask] = clipping_color
            else:
                min_color = np.array([0, 0, 0])
                max_color = np.array([255, 255, 255])
                arr[upper_clipping_mask] = max_color
                arr[lower_clipping_mask] = min_color

    img = _to_img(arr=arr, mode=out_format['mode'])

    min_color = "black" if not colorize else _cmap_min_str(cmap_name)
    max_color = "white" if not colorize else _cmap_max_str(cmap_name)
    if image_range_colors_off:
        image_range_text = "Image: Constant: %0.3f" % min_value if is_constant else "Min: %0.3f Max: %0.3f" % (arr_min, arr_max)
    else:
        image_range_text = "Image: Constant: %0.3f" % min_value if is_constant else "Min (%s): %0.3f Max (%s): %0.3f" % (min_color, arr_min, max_color, arr_max)

    draw_text = _get_draw_text(text, label, text_off, image_range_text, image_range_text_off)
    img = add_text_to_img(img=img, text=draw_text, xy_leftbottom=(5, 5))

    out = _convert_to_out_format(img, out_format)
    _apply_out_action(out=out, out_action=out_action, out_format=out_format)

    return out


def vis_image(img, full_batch=False, batch_labels=None, **kwargs):
    """
    Creates a visualization of an image in form of a numpy array or torch tensor.

    Args:
        img: Image in form of a numpy array or torch tensor.
        full_batch: Indicates whether all samples in the batch should be visualized.
            False: visualize only first sample in the batch.
            True/"cols": visualize all samples in the batch by concatenating col-wise (side-by-side).
            "rows": visualize all samples in the batch by concatenating row-wise.
        kwargs: See _vis_single_image function.
    """

    assert 3 <= img.ndim <= 4, f"Image array must have 3 or 4 dimensions, but got shape {img.shape}"
    if img.ndim == 3:
        assert img.shape[0] == 3, f"First dimension in a image array with shape {img.shape} must " \
                                  f"be 3, but got {img.shape[0]}."
    if img.ndim == 4:
        assert img.shape[1] == 3, f"Second dimension in a image array with shape {img.shape} must " \
                                  f"be 3, but got {img.shape[1]}."

    img = make_np(img)

    if full_batch:
        img = img[None, ...] if img.ndim == 3 else img
        imgs = []
        for idx, ele in enumerate(img):
            if batch_labels is not None:
                assert "label" not in kwargs, "It is not possible to use batch_labels and label argument at the same time."
                img_vis = _vis_single_image(ele, label=batch_labels[idx], **kwargs)
            else:
                img_vis = _vis_single_image(ele, **kwargs)
            imgs.append(img_vis)

        if full_batch == "rows":
            return cat_images_rowwise(imgs)
        else:
            return cat_images_colwise(imgs)

    else:
        img = img[0] if img.ndim == 4 else img
        return _vis_single_image(img, **kwargs)


def _vis_single_image(img,
                      clipping=False, upper_clipping_thresh=None, lower_clipping_thresh=None,
                      mark_clipping=False, clipping_color=None,
                      invalid_values=None, mark_invalid=False, invalid_color=None,
                      text=None, label=None, image_range_text_off=False, image_range_colors_off=False, text_off=False,
                      out_format=None, out_action=None):
    """
    Creates a visualization of a 2d numpy array or torch tensor.

    Args:
        img: 2D numpy array or torch tensor.
        colorize: If set to true, the values will be visualized by a colormap, otherwise as gray-values.
        clipping: If true, values above a certain threshold will be clipped before the visualization.
        upper_clipping_thresh: Threshold that is used for clipping the values. If set to False,
        the value mean + 2*std_deviation of the array will be used as threshold. The thresholds are also used
        as limits of the color range.
        lower_clipping_thresh: Threshold that is used for clipping the values. If set to False,
        the value mean - 2*std_deviation of the array will be used as threshold. The thresholds are also used
        as limits of the color range.
        mark_clipping: Mark clipped values with specific colors in the visualization.
        clipping_color: Color for marking clipped values.
        invalid_values: list of values that are invalid (e.g. [0]). If no such values exist, just pass None.
        mark_invalid: Mark invalid (NaN/Inf and all values in the invalid_values list) with
        specific colors in the visualization.
        invalid_color: Color for marking invalid values.
        text: Additional text that is printed on the visualization.
        cmap: Colormap to use for the visualization.
        desc=description string, colors=dict with keys marker_color, text_color, bbox_color, bbox_stroke, score).
        Everything except for coordinates can be None.
        image_range_text_off: If True, no text information about the range of the image values is added.
        text_off: If True, the provided text is not added to the image.
        out_format: Dict that describes the format of the output. All such dicts must have 'type' and 'mode' key.
        Currently supported are:
        {'type': 'PIL', 'mode': 'RGB' (see PIL docs for supported modes)} (this is the default format),
        {'type': 'np', 'mode': 'RGB' (see PIL docs for supported modes), 'dtype': 'uint8'}.
        out_action: Dict that describes an action on the output. All such dicts must have 'type' key.
        Note that some actions require a specific out_format.
        Currently supported are:
        None,
        {'type': 'show'}.
    """
    assert img.ndim == 3, f"Single image array must have 3 dimension, but got shape {img.shape}"
    img = make_np(img)

    img = img.astype(np.float32, copy=True).transpose(1, 2, 0)
    out_format = {'type': 'PIL', 'mode': 'RGB'} if out_format is None else out_format
    out_format['mode'] = 'RGB' if 'mode' not in out_format else out_format['mode']

    # Filter out all values that are somehow invalid and set them to 0:
    img, invalid_mask, invalid_values_mask, clipping_mask, upper_clipping_mask, lower_clipping_mask, \
    upper_clipping_thresh, lower_clipping_thresh = \
        invalidate_np_array(img, clipping, upper_clipping_thresh, lower_clipping_thresh, invalid_values)

    # Now work only with valid values of the array and make them visualizable (range 0, 256):
    arr_valid_only = np.ma.masked_array(img, invalid_mask)

    if not clipping:
        min_value = arr_min = float(np.ma.min(arr_valid_only))
        max_value = arr_max = float(np.ma.max(arr_valid_only))
    else:
        min_value = float(lower_clipping_thresh)
        max_value = float(upper_clipping_thresh)
        arr_min = float(np.ma.min(arr_valid_only))
        arr_max = float(np.ma.max(arr_valid_only))

    min_max_diff = max_value - min_value
    is_constant = (max_value == min_value)

    if is_constant:
        if min_value == 0:  # array is constant 0
            arr_valid_only *= 0
        else:
            arr_valid_only /= min_value
            arr_valid_only *= 255.0
    else:
        arr_valid_only -= min_value
        arr_valid_only /= min_max_diff
        arr_valid_only *= 255.0

    img = img.astype(np.uint8)

    if mark_invalid:
        invalid_color = np.array([0, 0, 0]) if invalid_color is None else invalid_color
        img[np.any(invalid_values_mask, axis=2)] = invalid_color

    if clipping:
        if mark_clipping:
            clipping_color = np.array([255, 255, 255]) if clipping_color is None else clipping_color
            img[np.any(clipping_mask, axis=2)] = clipping_color
        else:
            min_color = np.array([min_value] * 3)
            max_color = np.array([max_value] * 3)
            img[np.any(upper_clipping_mask, axis=2)] = max_color
            img[np.any(lower_clipping_mask, axis=2)] = min_color

    img = _to_img(arr=img, mode=out_format['mode'])

    image_range_text = "Image: Constant: %0.3f" % min_value if is_constant else "Min: %0.3f Max: %0.3f" % (
            arr_min, arr_max)

    draw_text = _get_draw_text(text, label, text_off, image_range_text, image_range_text_off)
    img = add_text_to_img(img=img, text=draw_text, xy_leftbottom=(5, 5))

    out = _convert_to_out_format(img, out_format)
    _apply_out_action(out=out, out_action=out_action, out_format=out_format)

    return out


def add_text_to_img(img, text,
                    xy_lefttop=None, xy_leftbottom=None,
                    x_abs_shift=None, y_abs_shift=None, x_rel_shift=None, y_rel_shift=None,
                    do_resize=True, resize_xy=False, max_resize_factor=None,
                    text_color=None, font=None, font_size=None,
                    bbox_color=_DEFAULT_BBOX_COLOR, bbox_stroke=_DEFAULT_BBOX_STROKE):
    """
    Add a text, optionally in a bounding box, to a PIL image.

    Upscales the image to fit the text if necessary. Note: in this case, a copy of the image is returned!

    Args:
        img: Image.
        text: Text. Can span multiple lines via '\n'.
        xy_lefttop: (x,y)-coordinate of the top-left-corner of the text. x is distance to left border, y to top.
        Either xy_lefttop or xy_leftbottom must not be None.
        xy_leftbottom: (x,y)-coordinate of the bottom-left-corner of the text. x is distance to left border, y to bottom.
        Either xy_lefttop or xy_leftbottom must not be None.
        x_abs_shift: Absolute offset in x direction to shift the text.
        y_abs_shift: Absolute offset in y direction to shift the text.
        x_rel_shift: Relative offset in x direction to shift the text.
        y_rel_shift: Relative offset in y direction to shift the text.
        do_resize: Specifies whether the image should be resized to fit the text.
        resize_xy: Specifies whether the (x,y)-position should be adjusted to the resize.
        max_resize_factor: Specifies the maximum factor for resizing the image.
        font: Font to be used for the text. If None, the default font will be used.
        font_size: Font size. If font is supplied, this parameter is ignored.
        text_color: (r,g,b) color for the text. If not set, default will be used.
        bbox_color: (r,g,b) color for a bbox to be drawn around the text. If not set, default will be used.
        bbox_stroke: (r,g,b) color for a bbox to be drawn around the text. If not set, default will be used.
    """

    if text == "":
        return img

    text_color = _DEFAULT_TEXT_COLOR if text_color is None else text_color
    font = _get_default_font(size=font_size) if font is None else font
    draw = ImageDraw.Draw(img)
    # text_size = draw.multiline_textsize(text=text, font=font)  # (width, height)
    if hasattr(draw, 'multiline_textbbox'):
        bbox = draw.multiline_textbbox((0, 0), text=text, font=font)
        text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    else:
        # fallback for older Pillow
        lines = text.split('\n')
        widths, heights = zip(*(draw.textsize(line, font=font) for line in lines))
        text_size = (max(widths), sum(heights))


    # shift xy pos according to xy_abs/rel_shifts:
    x_shift = (x_rel_shift * text_size[0] if x_rel_shift is not None else 0) + (x_abs_shift if x_abs_shift is not None else 0)
    y_shift = (y_rel_shift * text_size[1] if y_rel_shift is not None else 0) + (y_abs_shift if y_abs_shift is not None else 0)

    if xy_lefttop is not None:
        xy_lefttop = (xy_lefttop[0] + x_shift, xy_lefttop[1] + y_shift)

    if xy_leftbottom is not None:
        xy_leftbottom = (xy_leftbottom[0] + x_shift, xy_leftbottom[1] + y_shift)

    resized = False
    if do_resize:
        resize_factor = 1.0
        if xy_lefttop is not None:
            while img.width < text_size[0] + xy_lefttop[0] or img.height < text_size[1] + xy_lefttop[1]:

                if max_resize_factor is not None and resize_factor * 2 > max_resize_factor:
                    break

                img = img.resize(size=(img.width * 2, img.height * 2), resample=Image.NEAREST)

                xy_lefttop = (xy_lefttop[0] * 2, xy_lefttop[1] * 2) if resize_xy else xy_lefttop

                resize_factor *= 2
                resized = True
        else:
            while img.width < text_size[0] + xy_leftbottom[0] or img.height < text_size[1] + xy_leftbottom[1]:

                if max_resize_factor is not None and resize_factor * 2 > max_resize_factor:
                    break

                img = img.resize(size=(img.width * 2, img.height * 2), resample=Image.NEAREST)

                xy_leftbottom = (xy_leftbottom[0] * 2, xy_leftbottom[1] * 2) if resize_xy else xy_leftbottom

                resize_factor *= 2
                resized = True

    if xy_lefttop is None:
        xy_lefttop = (xy_leftbottom[0], img.height - xy_leftbottom[1] - text_size[1])

    draw = ImageDraw.Draw(img) if resized else draw

    if bbox_color is not None or bbox_stroke is not None:
        bbox_space = text_size[1] * 0.1
        # bbox = ([(xy_lefttop[0] - bbox_space, xy_lefttop[1] - bbox_space), (text_size[0] + xy_lefttop[0] + bbox_space + 1, text_size[1] + xy_lefttop[1] + bbox_space + 1)])
        # removed bbox space from top because somehow the text size estimates seemed to be slightly off anyways
        bbox = ([(xy_lefttop[0] - bbox_space, xy_lefttop[1]), (text_size[0] + xy_lefttop[0] + bbox_space + 1, text_size[1] + xy_lefttop[1] + bbox_space + 1)])
        draw.rectangle(bbox, bbox_color, bbox_stroke)

    draw.multiline_text(xy=xy_lefttop, text=text, fill=text_color, font=font)

    return img


def invalidate_np_array(arr, clipping=False, upper_clipping_thresh=None, lower_clipping_thresh=None, invalid_values=None):
    """
    Sets non-finite values (inf / nan), values that should be clipped (above / below some threshold), and specific values to 0.

    Can be used with arrays of arbitrary shapes. However, all filtering performs on single values only. So, for filtering
    values across multiple channels you have to split the array and filter each channel separately.
    """
    invalid_values_mask = np.isinf(arr) | np.isnan(arr)
    if invalid_values is not None:
        invalid_values_mask = invalid_values_mask | np.isin(arr, invalid_values)

    if clipping:
        if upper_clipping_thresh is None or lower_clipping_thresh is None:
            mean = np.nanmean(arr[~invalid_values_mask])
            std = np.nanstd(arr[~invalid_values_mask])
            all_values_invalid = np.all(invalid_values_mask)

            if upper_clipping_thresh is None:
                upper_clipping_thresh = min(np.nanmax(arr[~invalid_values_mask]), mean + 2 * std) if not all_values_invalid else np.nan
            if lower_clipping_thresh is None:
                lower_clipping_thresh = max(np.nanmin(arr[~invalid_values_mask]), mean - 2 * std) if not all_values_invalid else np.nan

        with np.errstate(invalid='ignore'):
            upper_clipping_mask = np.logical_and((arr > upper_clipping_thresh), ~invalid_values_mask)
            lower_clipping_mask = np.logical_and((arr < lower_clipping_thresh), ~invalid_values_mask)
        clipping_mask = upper_clipping_mask | lower_clipping_mask  # True = value should be clipped
    else:
        clipping_mask = np.zeros_like(arr, dtype='bool')  # All False because no values should be clipped
        upper_clipping_mask = clipping_mask
        lower_clipping_mask = clipping_mask

    invalid_mask = invalid_values_mask | clipping_mask
    arr[invalid_mask] = 0

    return arr, invalid_mask, invalid_values_mask, clipping_mask, upper_clipping_mask, lower_clipping_mask, upper_clipping_thresh, lower_clipping_thresh

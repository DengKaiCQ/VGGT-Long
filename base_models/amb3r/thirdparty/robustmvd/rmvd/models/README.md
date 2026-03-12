# Models

`rmvd` contains implementations of depth estimation models. The following provides an overview of the available models
and describes the usage of these models.

---

## Overview
`rmvd` contains two types of model implementations:
### 1. Native models:
Models that are (re-)implemented natively within the `rmvd` framework. 
These models can be used for training and inference/evaluation.

### 2. Wrapped models:
Model wrappers around existing implementations. These models can only be used for inference/evaluation.
Wrapped models are indicated by names that end with `_wrapped`.
The setup of these models is usually a bit more involved, as it is required to download the original implementation
(e.g. cloning the respective repository from GitHub). Usually the setup requires the follwing steps:
- clone the original repository to a local directory
- specify the path to the local directory in the `wrappers/paths.toml` file
- install required model-specific dependencies

The following provides an overview of all available models including their respective setup instructions.

---

## Available models

### `robust_mvd`
This is the Robust MVD Baseline Model presented in the publication 
"A Benchmark and a Baseline for Robust Depth Estimation" by Schröppel et al.

### `robust_mvd_5M`
This is the Robust MVD Baseline Model presented in the publication 
"A Benchmark and a Baseline for Robust Depth Estimation" by Schröppel et al., but trained for 5M iterations instead
of the 600k iterations in the paper. The longer training slightly improves results.

### `monodepth2_mono_stereo_1024x320_wrapped`
This is the "Monodepth2 (1024x320)" model presented in the publication 
"Digging into Self-Supervised Monocular Depth Estimation" by Godard et al. 
The model is wrapped around the original implementation from <https://github.com/nianticlabs/monodepth2>, where it is 
indicated as `mono+stereo_1024x320`.

#### Setup:
From the directory of this `README` file, execute the script `scripts/setup_monodepth2.sh` and specify the local
directory to clone the original repository:
```bash
./scripts/setup_monodepth2.sh /path/to/monodepth2
```

Then specify the local directory `/path/to/monodepth2` in the `wrappers/paths.toml` file (relative to the directory of  
this `README`).

It is not necessary to install additional dependencies.

#### Misc:
The model is applied at a fixed input size of `width=1024` and `height=320`. It therefore does not make sense to load
data at a specific downsampled resolution. Thus, don't use the `input_size` parameters of `Dataset` classes and of the
`eval.py` and `inference.py` scripts, when using this model.


### `monodepth2_mono_stereo_640x192_wrapped`
This is the "Monodepth2" model presented in the publication 
"Digging into Self-Supervised Monocular Depth Estimation" by Godard et al. 
The model is wrapped around the original implementation from <https://github.com/nianticlabs/monodepth2>, where it is 
indicated as `mono+stereo_640x192`.

#### Setup:
Same as for the `monodepth2_mono_stereo_1024x320_wrapped` model.

#### Misc:
The model is applied at a fixed input size of `width=640` and `height=192`. It therefore does not make sense to load
data at a specific downsampled resolution. Thus, don't use the `input_size` parameters of `Dataset` classes and of the
`eval.py` and `inference.py` scripts, when using this model.

### `mvsnet_pl_wrapped`
This is an unofficial implementation of the MVSNet model presented in the publication 
"MVSNet: Depth Inference for Unstructured Multi-view Stereo" by Yao et al. 
The model is wrapped around the unofficial implementation from <https://github.com/kwea123/MVSNet_pl>.

#### Setup:
From the directory of this `README` file, execute the script `scripts/setup_mvsnet_pl.sh` and specify the local
directory to clone the original repository:
```bash
./scripts/setup_mvsnet_pl.sh /path/to/mvsnet_pl
```

Then specify the local directory `/path/to/mvsnet_pl` in the `wrappers/paths.toml` file (relative to the directory of  
this `README`).

It is required to install additional dependencies. You might want to set up a new virtual environment for this:
```bash
pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.11
pip install kornia
```

### `midas_big_v2_1_wrapped`
This is the "MiDaS" model presented in the publication 
"Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer" by Ranftl et al. 
The model is wrapped around the original implementation from <https://github.com/isl-org/MiDaS>, where it is 
indicated as `Big models: MiDaS v2.1`.

#### Setup:
From the directory of this `README` file, execute the script `scripts/setup_midas.sh` and specify the local
directory to clone the original repository:
```bash
./scripts/setup_midas.sh /path/to/midas
```

Then specify the local directory `/path/to/midas` in the `wrappers/paths.toml` file (relative to the directory of  
this `README`).

It is not necessary to install additional dependencies.

### `vis_mvsnet_wrapped`
This is the Vis-MVSNet model presented in the publication "Visibility-aware Multi-view Stereo Network" by Zhang et al.
The model is wrapped around the original implementation from <https://github.com/jzhangbs/Vis-MVSNet.git>.

#### Setup:
From the directory of this `README` file, execute the script `scripts/setup_vis_mvsnet.sh` and specify the local
directory to clone the original repository:
```bash
./scripts/setup_vis_mvsnet.sh /path/to/vis_mvsnet
```

Then specify the local directory `/path/to/vis_mvsnet` in the `wrappers/paths.toml` file (relative to the directory of  
this `README`).

It is not necessary to install additional dependencies.

### `cvp_mvsnet_wrapped`
This is the CVP-MVSNet model presented in the publication 
"Cost Volume Pyramid Based Depth Inference for Multi-View Stereo" by Yang et al.
The model is wrapped around the original implementation from <https://github.com/JiayuYANG/CVP-MVSNet>.

#### Setup:
From the directory of this `README` file, execute the script `scripts/setup_cvp_mvsnet.sh` and specify the local
directory to clone the original repository:
```bash
./scripts/setup_cvp_mvsnet.sh /path/to/cvp_mvsnet
```

Then specify the local directory `/path/to/cvp_mvsnet` in the `wrappers/paths.toml` file (relative to the directory of  
this `README`).

It is not necessary to install additional dependencies.

#### Misc:
With the original implementation, the number of calculated depth hypotheses is sometimes too small. We apply a small
patch (see `cvp_mvsnet.patch` file) to fix this.

Further, the implementation does not support running the model with a single source view. It is therefore not possible
to evaluate the model with the `quasi-optimal` view selection, but only with the `nearest` view selection strategy.

### `patchmatchnet_wrapped`
This is the PatchmatchNet model presented in the publication 
"PatchmatchNet: Learned Multi-View Patchmatch Stereo" by Wang et al.
The model is wrapped around the original implementation from <https://github.com/FangjinhuaWang/PatchmatchNet>.

#### Setup:
From the directory of this `README` file, execute the script `scripts/setup_patchmatchnet.sh` and specify the local
directory to clone the original repository:
```bash
./scripts/setup_patchmatchnet.sh /path/to/patchmatchnet
```

Then specify the local directory `/path/to/patchmatchnet` in the `wrappers/paths.toml` file (relative to the directory 
of this `README`).

It is not necessary to install additional dependencies.

---

## Usage

All models can be used with the same interface. The following describes the usage of the models.

### Initialization

To initialize a model, use the `create_model` function:
```python
from rmvd import create_model
model_name = "robust_mvd"  # available models: see above (e.g. "monodepth2_mono_stereo_1024x320_wrapped", etc.)
model = create_model(model_name, pretrained=True, weights=None, train=False, num_gpus=1)  # optional: model-specific parameters
```

#### Weights

If `pretrained` is set to True, the default pretrained weights for the model will be used. 
Alternatively, custom weights can be loaded by providing the path to the weights with the `weights` parameter.

#### Train mode

If `train` is set to True, the model is created in training mode.

#### GPU usage

If `num_gpus` is `>0`, the model will be executed on the GPU.

### Inference
The interface to do inference with the model is:
```python
pred, aux = model.run(images=images, keyview_idx=keyview_idx, poses=poses, intrinsics=intrinsics, 
                      depth_range=depth_range)  # alternatively: run(**sample)
```

#### Inputs
The inputs can be:
- numpy arrays with a prepended batch dimension  (e.g. images are `N3HW` and of type `np.ndarray`)
- numpy arrays without a batch dimension (e.g. images are `3HW` and of type `np.ndarray`)

The formats of specific inputs are described in the [data readme](../data/README.md).

#### Outputs
The `pred` output is a dictionary which contains:
- `depth`: predicted depth map for the reference view
- `depth_uncertainty`: predicted uncertainty for the predicted depth map (optional)

The output type and shapes correspond to the input types and shapes, i.e.:
- numpy arrays with a prepended batch dimension  (e.g. `depth` has shape `N1HW` and type `np.ndarray`)
- numpy arrays without a batch dimension (e.g. `depth` has shape `1HW` and type `np.ndarray`)

The `aux` output is a dictionary which contains additional, model-specific outputs. These are only used for training 
or debugging and not further described here.

#### Resolution
Most models cannot handle input images at arbitrary resolutions. Models therefore internally upsize the images to
the next resolution that can be handled. 

The model output is often at a lower resolution as the input data.

---

## Internal implementation

Internally, all models have the following functions:
- a `input_adapter` function that converts input data into the models-specific format
- a `forward` function that runs a forward pass with the model (in non-pytorch models, this is the `__call__` function) 
- a `output_adapter` function that converts predictions from model-specific format to the `rmvd` format

The `run` function mentioned above to do inference, uses those three functions as follows:
```python
def run(images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
    no_batch_dim = (images[0].ndim == 3)
    if no_batch_dim:
        images, keyview_idx, poses, intrinsics, depth_range = \
            add_batch_dim(images, keyview_idx, poses, intrinsics, depth_range)

    sample = model.input_adapter(images=images, keyview_idx=keyview_idx, poses=poses,
                                 intrinsics=intrinsics, depth_range=depth_range)
    model_output = model(**sample)
    pred, aux = model.output_adapter(model_output)

    if no_batch_dim:
        pred, aux = remove_batch_dim(pred, aux)

    return pred, aux
```

In the following, we further describe the `input_adapter`, `forward`/`__call__` and `output_adapter` functions.

### The `input_adapter` function

The `input_adapter` function has the following interface: 
```python
def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
    # construct sample dict that contains all inputs in the model-specific format: sample = {..}
    return sample
```
The inputs to the `input_adapter` function are all `numpy` array with a batch dimension 
(e.g. images are `N3HW` and of type `np.ndarray`). The function then converts all inputs to the format that
is required by the model and returns this converted data as a dictionary where the keys are the parameter names
of the model's `forward` / `__call__` function. This allows to call `model(**sample)` where sample is the dictionary 
that is returned from the `input_adapter` function.

The conversion may for example include converting the inputs to `torch.Tensor`, moving them to the GPU if required, 
normalizing the images, etc.

### The `forward` function (for non-pytorch model this function is named `__call__`)
The `forward` function of each model expects data in the model-specific format and returns model-specific outputs.

Hence, in case all input data is already in the format required by the model, you can also do `model(**sample)`. 
This is used in the `rmvd` training code. Note that the `forward` function also expects input data to have a resolution
that is supported by the model.

### The `output_adapter` function
The `output_adapter` function has the following interface:
```python
def output_adapter(self, model_output):
    # construct pred and aux dicts from model_output
    # pred needs to have an item with key "depth" and value of type np.ndarray and shape N1HW
    return pred, aux
```
The output adapter converts model-specific outputs to the `pred` and `aux` dictionaries. The output types and shapes
need to be numpy arrays with a batch dimension (i.e. `depth` has shape `N1HW` and type `np.ndarray`). 

---

## Using custom models within the `rmvd` framework

If you want to use your own model within the framework, e.g. for evaluation, your model needs to have the
`input_adapter`, `forward`/`__call__` and `output_adapter` functions as described above.

Note: you don't have to add a `run` function your model. This function will be added automatically by calling
`rmvd.prepare_custom_model(model)`.

You can then use your custom model within the `rmvd` framework, for example to run inference, e.g.:
```python
import rmvd
model = CustomModel()
model = rmvd.prepare_custom_model(model)
dataset = rmvd.create_dataset("eth3d", "mvd", input_size=(384, 576))
sample = dataset[0]
pred, aux = model.run(**sample)
```
or to run evaluation, e.g.:
```python
import rmvd
model = CustomModel()
model = rmvd.prepare_custom_model(model)
eval = rmvd.create_evaluation(evaluation_type="mvd", out_dir="/tmp/eval_output", inputs=["intrinsics", "poses"])
dataset = rmvd.create_dataset("kitti", "mvd", input_size=(384, 1280))
results = eval(dataset=dataset, model=model)
```

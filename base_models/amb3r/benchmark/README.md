# AMB3R Benchmark

This is the benchmark that supplies AMB3R paper on various 3D reconstruction tasks.

```bibtex
@article{wang2025amb3r,
  title={AMB3R: Accurate Feed-forward Metric-scale 3D Reconstruction with Backend},
  author={Wang, Hengyi and Agapito, Lourdes},
  journal={arXiv preprint arXiv:2511.20343},
  year={2025}
}
```

## Table of Contents
- [Monocular Depth Estimation](#monocular-depth-estimation)
- [Camera Pose Estimation](#camera-pose-estimation)
- [Multi-view Depth Estimation](#multi-view-depth-estimation)
- [Video Depth Estimation](#video-depth-estimation)
- [Multi-view 3D Reconstruction](#multi-view-3d-reconstruction)
- [Visual Odometry/SLAM](#visual-odometryslam)
- [Structure from Motion](#structure-from-motion)
- [Evaluating Custom Models](#evaluating-custom-models)

## Monocular Depth Estimation


### Data Preparation
We follow Marigold and Diffusion-E2E and evaluate on the following datasets: **NYUv2**, **KITTI**, **ETH3D**, **ScanNet**, and **DIODE**. You can download the datasets by running the following script:

```sh
cd ../scripts
bash download_mono.sh
cd ../benchmarks
```

### Evaluation
```bash
python eval_monodepth.py
```



## Camera Pose Estimation

Following VGGT, we evaluate on the RealEstate10K dataset. Due to the limited availability of some YouTube videos in RealEstate10K, we provide a pre-processed evaluation split: [re10k_amb3r_split](https://huggingface.co/datasets/HengyiWang/re10k_amb3r_split). This split is generated using the same random frame sampling strategy described in VGGT, enabling the community to reproduce the evaluation on RealEstate10K.

> **Note:** The released split contains 1,721 sequences, each consisting of 10 frames randomly sampled from the corresponding full video. As this split is not exactly identical to the one originally used in VGGT, we kindly ask that you explicitly state the use of the `re10k_amb3r_split` in your paper to ensure fair comparison and reproducibility.

```sh
cd ../scripts
bash download_pose.sh
cd ../benchmarks
```

### Evaluation
```bash
python eval_pose.py
```


## Multi-view Depth Estimation

### Data Preparation
Following RMVDB, we evaluate using the **Tanks and Temples**, **ETH3D**, **ScanNet**, **DTU**, and **KITTI** datasets.

```sh
cd ../scripts
bash download_rmvd.sh
cd ../benchmarks
```

> **Note:** **ScanNet** and **KITTI** datasets must be downloaded manually.

### Evaluation
To evaluate metric-scale depth, append the `--metric` flag:
```bash
python eval_mvdepth.py
```

---

## Video Depth Estimation

### Data Preparation
We use the **Sintel**, **Bonn**, and **KITTI** datasets for evaluation.

```sh
cd ../scripts
bash download_videodepth.sh
cd ../benchmarks
```

### Evaluation
> **Note:** Evaluating on the Bonn and KITTI dataset would require a GPU with more than 24GB of memory.

```bash
python eval_videodepth.py
```

---

## Multi-view 3D Reconstruction

### Data Preparation
We evaluate on the **ETH3D**, **DTU**, and **7Scenes** datasets.
- For ETH3D and DTU, image tuples are from RMVDB.
- For 7Scenes, image tuples are from Spann3R.

```sh
cd ../scripts
bash download_3d.sh
cd ../benchmarks
```

> **Note:** As the original 7Scenes dataset contains imperfect poses for evaluation, we provide pre-processed poses [here](./data/7scenes_sfm_poses.zip). These are COLMAP poses provided by [visloc_pseudo_gt](https://github.com/tsattler/visloc_pseudo_gt_limitations). The preparation script will automatically extract and place these processed poses in the correct directory.

### Evaluation
```bash
python eval_mvrecon.py
```

---

## Visual Odometry / SLAM

### Data Preparation
We use the **TUM**, **ETH3D SLAM**, and **7Scenes** datasets for SLAM evaluation.

```sh
cd ../scripts
bash download_slam.sh
cd ../benchmarks
```

### Evaluation
```bash
python eval_slam.py
```

---

## Structure from Motion (SfM)

### Data Preparation
We evaluate on the **ETH3D**, **Tanks and Temples (TnT)**, and **IMC Phototourism** datasets.
ETH3D is downloaded during the Multi-view Depth Estimation setup. Use the following script to download TnT and IMC:

```sh
cd ../scripts
bash download_sfm.sh
cd ../benchmarks
```

### Evaluation
```bash
python eval_sfm.py
```

---

## Evaluating Custom Models

You can easily plug in and evaluate your own models with AMB3R-Benchmark.

For **Monocular Depth Estimation**, **Camera Pose Estimation**, **Multi-view 3D Reconstruction**, and **Video Depth Estimation**, implement a `run_amb3r_benchmark` method in your model class:

```python
def run_amb3r_benchmark(self, frames):
    # frames['images']: (B, T, C, H, W) normalized in [-1, 1]
    images = frames['images'] 
    
    # Run your own base model to get pointmap, pose, and confidence
    pointmap, pose, confidence, pts3d_by_unprojection = self.forward(images)

    return {
        'world_points': pointmap,            # Pointmaps
        'depth': depth_map,                  # Depth predictions
        'pose': pose,                        # Camera poses
        'pts3d_by_unprojection': pts3d       # Unprojected 3D points
    }
```

For **Multi-view Depth Estimation**, you'll need to write custom input and output adapter functions, following the structure documented [here](https://github.com/lmb-freiburg/robustmvd/blob/master/rmvd/models/README.md).


For **Visual Odometry** and **Structure-from-Motion**, please implement the following methods in your model:
- `run_amb3r_vo`
- `run_amb3r_sfm`


## Citation

If you find our code, data, or paper useful, please consider citing:

```bibtex
@article{wang2025amb3r,
  title={AMB3R: Accurate Feed-forward Metric-scale 3D Reconstruction with Backend},
  author={Wang, Hengyi and Agapito, Lourdes},
  journal={arXiv preprint arXiv:2511.20343},
  year={2025}
}
```


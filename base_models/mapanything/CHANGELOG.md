# Changelog

All notable changes to MapAnything will be documented in this file.

## [1.0.0] - 2025-09-15

### Added
- Initial public release of MapAnything: Universal Feed-Forward Metric 3D Reconstruction
- Complete codebase for inference, data processing, benchmarking, training, and ablations
- Two pre-trained model variants on Hugging Face Hub:
  - `facebook/map-anything` (CC-BY-NC 4.0 License) - Research & Academic Use
  - `facebook/map-anything-apache` (Apache 2.0 License) - Commercial Use
- Image-only inference support for metric 3D reconstruction from images
- Multi-modal inference support with flexible combinations of:
  - Images + Camera intrinsics
  - Images + Depth maps
  - Images + Camera poses
  - Any combination of the above inputs
- Interactive demos:
  - Online Hugging Face demo
  - Local Gradio demo with GUI interface
  - Rerun demo for interactive 3D visualization
- COLMAP & GSplat integration:
  - Direct export to COLMAP format
  - Bundle adjustment support
  - Gaussian Splatting compatibility
- Comprehensive data processing pipeline for 13 training datasets
- Complete training framework with:
  - Memory optimization support
  - All main model and ablation training scripts
  - Fine-tuning support for other geometry estimation models
- Benchmarking suite with:
  - Dense Up-to-N-View Reconstruction Benchmark
  - Single-View Image Calibration Benchmark
  - RobustMVD Benchmark
- Building blocks for the community:
  - UniCeption library for modular network components
  - WorldAI (WAI) unified data format for 3D/4D/Spatial AI
- Apache 2.0 licensed codebase for open-source development
- Complete documentation with installation instructions, API reference, and examples

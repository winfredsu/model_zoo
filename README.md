# Model Zoo
This repo contains neural network models and training scripts that are friendly to FPGA/ASIC acceleration. 

# Structure
- `datasets`: example datasets and dataset utils
- `frameworks`: guides on training a new model
- `nets`: training scripts of various NN architectures

# Datasets
## Oxford-IIIT
For image classification, object detection (single object) and segmentation (single object)

# DL Frameworks
## TensorFlow 1.13.1
- From scratch
- Object-detection API

## TensorFlow 2.0
TBD. 

# Available Network Architectures
|NN Architectures|TensorFlow 1.13.1 with power-of-2 quant-aware-training|TensorFlow 2.x|
|-|:-:|-|
|Mobilenet-v1|$\bullet$|-|
|Mobilenet-v2-SSD|-|-|

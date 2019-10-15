# Mobilenet-V1-pets
This is a mobilenet-v1 trained on Oxford-IIIT dataset. The input image size is 160 by 160, and the depthwise multiplier is 0.25. 

## Scripts
- `config.py`: defines dataset and training related parameters.
- `dataset.py`: prepares the dataset using tf.data API.
- `model.py`: defines the mobilenet-v1 architecture. 
- `train.py`: training script.
- `freeze.py`: script to generate the inference graph.

## Usage
### 0. Prepare dataset
Modify `dataset.py` and `config.py` to use other datasets. 
### 1. Define the net
Modify `model.py` to use other network architectures. 
### 2. Train with FP
Train using `train.py` with quantization disabled, until getting a good accuracy. 
### 3. Quant-aware-training
Train using `train.py` with quantization enabled, and start from a FP checkpoint. 
### 4. Generate inference graph
Execute `freeze.py` to get the inference frozen graph. 


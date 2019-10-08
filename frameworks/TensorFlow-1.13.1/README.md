# Model Optimization in TensorFlow
This directory contains necessary patches to perform model optimization in TensorFlow 1.x. Before training a new model, you should have TensorFlow 1.13.1 installed. 

## Optimization Techniques
Currently, only power-of-2 quant-aware-training is supported. 
### Power-of-2 Quant-Aware-Training
Principles TBD.

By default, the weights, biases and activations are quantized to 8-bit integers. And `relu6` is the recommended activation function since it helps to limit the range of input tensor. 

## Patches
### `quant_ops.py`
Add `max_po2` and `min_po2` in `LastValueQuantize` and `MovingAvgQuantize`. 

### `quantize.py`
Add `bias_quant` to the patterns for inserting fake quant ops. The default decay of exponential moving average for bn_fold is 0.999. Change this value in the definition of `Quantize` function if you need. 

### `quantize_graph.py`
Call graph: `create_training/eval_graph` -> `_create_graph` -> `FoldBatchNorms` and `Quantize`. 

## Usage
### 1. Copy (and modify if needed) the patches
Replace `quant_ops.py`, `quantize.py` and `quantize_graph.py` in `PATH_TO_TENSORFLOW/contrib/quantize/python` with the files provided in `patches`. 

### 2. Create a model and train
#### 2.1 Train from scratch
TBD. 

#### 2.2 Use the object detection API
Follow the instructions in `object_detection/README.md`.

### 3. Check the frozen inference graph
In this step, we need to check if the optimization is successful. 

First, the loss should converge and you should get a desired validation accuracy. 

Second, you should see lots of `FakeQuantWithMinMaxVars` ops inserted in the inference graph. Specifically:
- For each **convolution** and **fully connect** layer, the input tensor, weight tensor and bias tensor (if present) should be (or from) a fake_quant op, and there should be a fake_quant op after the output tensor. The activation function (if present) should be `relu6` or `relu` (other activation functions are not supported by far). 
- For each **add** layer at the end of a residual block, both inputs should be fake_quant ops. And there should be a fake_quant op after the output tensor. 

`examples/example_mnetv2_0.5_ssd.pb` is an example of correctly trained, quantized and frozen graph. 


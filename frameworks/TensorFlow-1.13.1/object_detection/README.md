# The Object Detection API
Follow the below steps to use the object detection API to create and train a model in TensorFlow 1.13.1. Be sure to have the object detection API installed. [installation guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) 

## Patches
### `box_head.py`
This file defines box prediction heads for different meta architectures. We added `slim.batch_norm` as the normalizer_fn of `ConvolutionalBoxHead`, which is used by the SSD architecture. 

### `class_head.py`
This file defines class prediction heads for different meta architectures. We added `slim.batch_norm` as the normalizer_fn of `ConvolutionalClassHead`, which is used by the SSD architecture. 

## Usage
### 1. Config the pipeline
You can start from an example in `PATH_TO_OBJECT_DETECTION/samples/configs`. For example, if you want to train a mobilenet-v2-ssd detector, you can choose the `ssd_mobilenet_v2_quantized_300x300_coco.config` as a start point and modify the following items:
- `model->ssd->num_classes`: number of classes in the dataset (excluding the background)
- `model->ssd->image_resizer->fixed_shape_resizer`: width and height of the input image
- `model->ssd->box_predictor->convolutional_box_predictor`: decay of batchnorm
- `model->ssd->feature_extractor`: depth multiplier, decay of batchnorm
- `model->ssd->post_processing`: max detections
- `train_config->batch_size`: batch size
- `train_config->optimizer`: optimizer and learning rate
- `train_config->fine_tune_checkpoint/fine_tune_checkpoint_type`: config the path if you have a quantized checkpoint. If not, just comment these two lines. It's not recommended to start from a floating-point checkpoint. 
- `train_input_reader`: path to tfrecords of the training set and label map
- `eval_config`: number of images to eval. `max_evals` is recommended to be removed
- `eval_input_reader`: path to tfrecords of the eval set and label map
- `graph_rewriter`: quantization delay

### 2. Modify the detection heads
The quant-aware-training scripts can successfully insert fake_quant ops into the backbone net, but the detection heads (such as SSD) cannot be properly quantized. For example, if you train with `ssd_mobilenet_v2_quantized_300x300_coco.config`, the biases of the box predictors and class predictors will not be quantized. This is because our quant-aware-training method requires each conv/fc/add operation uses `slim.batch_norm` as the normalizer function. To make it work, you have to modify some codes in the detection heads. 

For example, `box_head.py` and `class_head.py` in the `patches` directory provide an example of how to modify the heads for SSD. If you are going to use SSD architecture, you can replace `box_head.py` and `class_head.py` in `PATH_TO_OBJECT_DETECTION/predictors/heads` with the files provided in `patches`. If you are using other detectors, you can modify the heads on your own. 

### 3. Train 
Start training with the following command:
``` bash
python PATH_TO_OBJECT_DETECTION/model_main.py --pipeline_config_path=PATH_TO_CONFIG_FILE_IN_STEP1 --model_dir=train/ --num_train_steps=1000000 --sample_1_of_n_eval_examples=1 --alsologtostderr
```

You can start a tensorboard in the training directory to see whether the loss converges. The quant-aware-training starts after the steps specified by `quantization_delay`. At this moment, you should see a slightly higher loss.

### 4. Freeze an inference graph
Export the frozen graph with the following command:
``` bash
python PATH_TO_OBJECT_DETECTION/export_tflite_ssd_graph.py --pipeline_config_path=PATH_TO_CONFIG_FILE_IN_STEP1 --trained_checkpoint_prefix=train/model.ckpt-XXXXXX --output_directory=inference --add_postprocessing=false
```

Note that this only works for SSD-like architectures. Supports for RCNN and other architectures will be provided soon. 

After this step, you can check if you've got a valid model, as described in `../README.md`.  

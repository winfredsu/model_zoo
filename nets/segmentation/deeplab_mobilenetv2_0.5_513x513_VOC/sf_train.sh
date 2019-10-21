CUDA_VISIBLE_DEVICES=3 python ~/Projects/tensorflow_models/models/research/deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=20000 \
    --train_split="train" \
    --model_variant="mobilenet_v2" \
    --depth_multiplier=0.5 \
    --output_stride=16 \
    --train_crop_size="513,513" \
    --train_batch_size=8 \
    --dataset="pascal_voc_seg" \
    --base_learning_rate=3e-4 \
    --tf_initial_checkpoint="./ckpt/model.ckpt" \
    --initialize_last_layer \
    --quantize_delay_step=10 \
    --train_logdir="./train" \
    --dataset_dir="/home/sufang/Projects/model_zoo/datasets/PASCAL_VOC_2012/tfrecord"

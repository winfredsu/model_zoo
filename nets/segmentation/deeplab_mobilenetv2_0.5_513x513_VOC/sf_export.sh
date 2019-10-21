python ~/Projects/tensorflow_models/models/research/deeplab/export_model.py \
    --checkpoint_path=train/model.ckpt-16571 \
    --depth_multiplier=0.5 \
    --quantize_delay_step=0 \
    --export_path=frozen_inference_graph.pb

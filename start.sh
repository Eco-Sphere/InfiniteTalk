MINDIESD_PATH=/usr/local/python3.11.14/lib/python3.11/site-packages/mindiesd
export ASCEND_CUSTOM_OPP_PATH=$MINDIESD_PATH/ops/vendors/customize:$MINDIESD_PATH/ops/vendors/aie_ascendc:

NPU_NUM=8
export HCCL_CONNECT_TIMEOUT=3600
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export TASK_QUEUE_ENABLE=2
export LD_PRELOAD=/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64/libjemalloc.so:$LD_PRELOAD
export CPU_AFFINITY_CONF=2

torchrun --nproc_per_node=$NPU_NUM --standalone generate_infinitetalk.py \
--ckpt_dir /data/z00823791/weight/Wan2.1-I2V-14B-480P \
--wav2vec_dir /data/z00823791/weight/chinese-wav2vec2-base \
--infinitetalk_dir /data/z00823791/weight/InfiniteTalk-single/single/infinitetalk.safetensors \
--ulysses_size=$NPU_NUM \
--input_json examples/single_example_image.json \
--size infinitetalk-480 \
--t5_fsdp \
--sample_steps 4 \
--lora_dir /data/z00823791/weight/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors \
--mode streaming \
--motion_frame 9 \
--sample_text_guide_scale 1.0 \
--sample_audio_guide_scale 1.0 \
--lora_scale 1.0 \
--sample_shift 11 \
--use_rainfusion \
--sparsity 0.85 \
--sparse_start_step 1 \
--rainfusion_type "v2" \
--save_file infinitetalk_sigle


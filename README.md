## 一、准备运行环境

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.11.14 | - |
  | torch | 2.9.0 | - |
  | CANN | 8.5.0 | - |

### 1.1 镜像下载
- 设备支持
Atlas 800I/800T A2(8*64G) 800I/800T A3(8*64G)推理设备：支持的卡数最小为A2 8卡或者A3 4卡
- [Atlas 800I/800T A2(8*64G)](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=netconda&OS=openEuler)

### 1.2 pip包依赖安装
```shell
pip3 install -r requirements_env.txt

pip install phonemizer-fork==3.3.2
```

### 1.3 环境依赖安装（long-context-attention、xDiT）
```shell
# 安装long-context-attention
git clone https://github.com/feifeibear/long-context-attention.git
cd long-context-attention/
pip install .

# 安装xDit
git clone https://github.com/xdit-project/xDiT.git
cd xDiT/
pip install -e .
```

### 1.4 decord安装
```shell
# 下载ffmpeg包
wget https://ffmpeg.org/releases/ffmpeg-4.2.1.tar.gz

# 安装ffmpeg
tar -zxvf ffmpeg-4.2.1.tar.gz
cd ffmpeg-4.2.1
./configure  --enable-shared --prefix=/usr/local/ffmpeg
make -j
make install
vi ~/.bashrc
export FFMPEG_PATH=/usr/local/ffmpeg/
export PATH=$FFMPEG_PATH/bin:$PATH
export LD_LIBRARY_PATH=$FFMPEG_PATH/lib:$LD_LIBRARY_PATH
source ~/.bashrc

# 安装decord
git clone --recursive https://github.com/dmlc/decord.git
cd decord
mkdir build && cd build
cmake .. -DFFMPEG_DIR=/usr/local/ffmpeg
make
cd ../python
pwd=$PWD
echo "PYTHONPATH=$PYTHONPATH:$pwd" >> ~/.bashrc
source ~/.bashrc
python3 setup.py install --user
```

### 1.5 MindIE-SD安装
```shell
#使用源码进行编译安装
git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD python setup.py bdist_wheel 
cd dist 
pip install mindiesd-*.whl
```

### 1.6 系统依赖安装
```shell
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### 1.7 网络配置
使用hostname获取当前主机名称，在/etc/hosts文件后追加配置
```shell
{本机IP}  {主机名}
```

## 二、下载权重

### 2.1 权重及配置文件说明
| Models        |                       Download Link                                           |    Notes                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.1-I2V-14B-480P  |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)       | Base model
| Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64      |      🤗 [Huggingface](https://huggingface.co/lgylgy/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64)              | wan lora weights
| chinese-wav2vec2-base |      🤗 [Huggingface](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base)          | Audio encoder
| MeiGen-InfiniteTalk      |      🤗 [Huggingface](https://huggingface.co/MeiGen-AI/InfiniteTalk)              | Our audio condition weights

Download models using huggingface-cli:
``` sh
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
huggingface-cli download lgylgy/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64 --local-dir ./weights/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk

```

## 三、InfiniteTalk使用

### 3.1 下载到本地
```shell
git clone https://github.com/Eco-Sphere/InfiniteTalk.git
```

### 3.2 性能测试（开启稀疏）

执行命令：
```shell
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
```
参数说明：
- NPU_NUM: 使用npu卡数
- size: 生成视频的尺寸
- sample_steps: 生成视频时执行的步数
- use_rainfusion: 开启LA稀疏
- sparsity: LA稀疏系数，值越大，精度损失越高
- sparse_start_step: 开启稀疏的步数
- rainfusion_type: 稀疏的版本，当前只支持V2

注：开启LA稀疏后，会有精度损失，LA稀疏系数越高，性能收益越高，精度损失越大，具体损失需要根据业务实测

### 3.3 性能测试

执行命令：
```shell
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
--save_file infinitetalk_sigle
```
参数说明：
- NPU_NUM: 使用npu卡数
- size: 生成视频的尺寸
- sample_steps: 生成视频时执行的步数

## 四、量化功能支持

在线量化wan2.1和lora权重后，精度损失较大，且性能收益较低，不建议进行量化

## 五、推理结果参考
  | 模型  | 硬件型号 | 卡数 | 分辨率 | LA稀疏 | 4步 E2E耗时（s）
  | ----- | ----- | ----- | -----| -----| -----|
  | InfiniteTalk | A2 910B3 | 8 | 480P | 关闭 | 82 |
  | InfiniteTalk | A2 910B3 | 8 | 480P | 0.85 | 78 |
  | InfiniteTalk | 800T A3 | 4 | 480P | 关闭 | 64 |
  | InfiniteTalk | 800T A3 | 4 | 480P | 0.85 | 59 |

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。

# python run_autoglm_v.py --provider_name vmware --path_to_vm D:\projects\OSWorld\vmware_vm_data\Ubuntu0\Ubuntu0.vmx --headless --max_steps 15 --test_all_meta_path ./evaluation_examples/test_one.json

# 部署LLM
git lfs install
git clone https://www.modelscope.cn/shawliu9/computerrl-glm4_1v-9b.git

export ENABLE_JIT_DEEPGEMM=0
export SGLANG_DISABLE_CUDNN_CHECK=1

python -m sglang.launch_server \
  --model-path /data1/lwm/projects/computerrl-glm4_1v-9b \
  --host 0.0.0.0 \
  --port 30000 \
  --served-model-name autoglm-os \
  --mem-fraction-static 0.4 \
  --disable-cuda-graph \
  --attention-backend triton

nohup python -m vllm.entrypoints.openai.api_server --served-model-name autoglm-os --model /data1/lwm/projects/computerrl-glm4_1v-9b --gpu-memory-utilization 0.4 --port 30000 > autoglm.log 2>&1 &

# 下载docker镜像
git lfs install
git clone https://huggingface.co/datasets/xlangai/ubuntu_osworld
cd ubuntu_osworld
unzip Ubuntu.qcow2.zip

# 加入docker用户组
sudo usermod -aG docker $USER
newgrp docker

# 运行
# export OPENAI_BASE_URL="http://localhost:30000/v1"
# export OPENAI_API_KEY="EMPTY"
conda activate spider2v
python run_autoglm_v.py \
    --provider_name docker \
    --path_to_vm /data1/lwm/projects/ubuntu_osworld/Ubuntu.qcow2 \
    --result_dir results/autoglm-os_baseline \
    --headless \
    --max_steps 100 \
    --test_all_meta_path ./evaluation_examples/test_all.json

# 杀进程
pkill -f run_autoglm_v.py
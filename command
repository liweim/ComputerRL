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

conda activate uitars
export CUDA_VISIBLE_DEVICES=0
nohup python -m vllm.entrypoints.openai.api_server --served-model-name autoglm-os --model /data1/lwm/projects/computerrl-glm4_1v-9b --gpu-memory-utilization 0.4 --port 30000 > autoglm.log 2>&1 &

# 启动UI-TARS
conda activate uitars
nohup python -m vllm.entrypoints.openai.api_server --served-model-name uitars-1.5-7b --model /data1/lwm/models/UI-TARS-1.5-7B --gpu-memory-utilization 0.4 --max-model-len 65536 --port 1235 > uitars.log 2>&1 &
nohup python -m vllm.entrypoints.openai.api_server --served-model-name gta1-7b --model /data1/lwm/models/GTA1-7B --gpu-memory-utilization 0.4 --max-model-len 65536 --port 1234 > gta1.log 2>&1 &

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
nohup \
python run_autoglm_v.py \
    --provider_name docker \
    --path_to_vm /data1/lwm/projects/ubuntu_osworld/Ubuntu.qcow2 \
    --result_dir results/autoglm-os_baseline \
    --headless \
    --max_steps 100 \
    --test_all_meta_path ./evaluation_examples/test_all.json \
    > nohup3.out 2>&1 &

nohup \
python run_autoglm_v_recovery.py \
    --provider_name docker \
    --path_to_vm /data1/lwm/projects/ubuntu_osworld/Ubuntu.qcow2 \
    --result_dir results/autoglm-os_gta1_7b_recovery \
    --visual_grounder_model gta1-7b \
    --headless \
    --max_steps 100 \
    --test_all_meta_path ./evaluation_examples/test_one.json \
    --rerun_fail \
    > nohup3.out 2>&1 &

nohup \
python run_hisa.py \
  --provider_name docker \
  --path_to_vm /data1/lwm/projects/ubuntu_osworld/Ubuntu.qcow2 \
  --result_dir ./results/hisa_gta1_7b_wo_step_refinement_pattern \
  --visual_grounder_model gta1-7b \
  --headless \
  --max_steps 100 \
  --test_all_meta_path ./evaluation_examples/test_one.json \
  --wo_step \
  --wo_refinement \
  --wo_pattern \
  > nohup.out 2>&1 &

# 杀进程
ps -ef | grep run_autoglm_v.py
pkill -f run_autoglm_v.py
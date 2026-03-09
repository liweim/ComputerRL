# python run_autoglm_v.py --provider_name vmware --path_to_vm D:\projects\OSWorld\vmware_vm_data\Ubuntu0\Ubuntu0.vmx --headless --max_steps 15 --test_all_meta_path ./evaluation_examples/test_one.json

# 部署LLM
git lfs install
git clone https://www.modelscope.cn/shawliu9/computerrl-glm4_1v-9b.git
git clone https://huggingface.co/Qwen/Qwen3.5-9B.git

export ENABLE_JIT_DEEPGEMM=0
export SGLANG_DISABLE_CUDNN_CHECK=1

conda activate uitars
export CUDA_VISIBLE_DEVICES=0
nohup python -m vllm.entrypoints.openai.api_server --served-model-name autoglm-os --model /data1/lwm/projects/computerrl-glm4_1v-9b --gpu-memory-utilization 0.4 --port 30000 > autoglm.log 2>&1 &

# 启动UI-TARS
conda activate uitars
nohup python -m vllm.entrypoints.openai.api_server --served-model-name uitars-1.5-7b --model /data1/lwm/models/UI-TARS-1.5-7B --gpu-memory-utilization 0.4 --max-model-len 65536 --port 1235 > uitars.log 2>&1 &
nohup python -m vllm.entrypoints.openai.api_server --served-model-name gta1-7b --model /data1/lwm/models/GTA1-7B --gpu-memory-utilization 0.4 --max-model-len 65536 --port 1234 > gta1.log 2>&1 &
nohup python -m vllm.entrypoints.openai.api_server --served-model-name qwen3.5-9b --model /data1/lwm/models/Qwen3.5-9B --gpu-memory-utilization 0.4 --max-model-len 65536 --port 30000 > qwen.log 2>&1 &


# 启动embedding service
conda activate uitars
nohup python /data1/lwm/projects/ComputerRL/mm_agents/autoglm_v_restart/embedding.py > embedding.log 2>&1 &

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
    > nohup.out 2>&1 &

nohup \
python run_autoglm_v_restart.py \
    --provider_name docker \
    --path_to_vm /data1/lwm/projects/ubuntu_osworld/Ubuntu.qcow2 \
    --result_dir results/autoglm-os_gta1_7b_restart_ori_res \
    --visual_grounder_model gta1-7b \
    --screen_width 1920 \
    --screen_height 1080 \
    --headless \
    --max_steps 100 \
    --test_all_meta_path ./evaluation_examples/test_small.json \
    > nohup3.out 2>&1 &

nohup \
python run_hisa.py \
  --provider_name docker \
  --path_to_vm /data1/lwm/projects/ubuntu_osworld/Ubuntu.qcow2 \
  --result_dir ./results/hisa_qwen3.5-9b_wo_step_refinement_pattern \
  --headless \
  --max_steps 100 \
  --test_all_meta_path ./evaluation_examples/debug.json \
  --wo_step \
  --wo_refinement \
  --wo_pattern \
  > nohup.out 2>&1 &

# 杀进程
ps -ef | grep run_autoglm_v_restart.py
pkill -f run_autoglm_v_restart.py
ps -fp 2011314
tr '\0' ' ' < /proc/2010759/cmdline ; echo
fuser -k 8001/tcp


# debug
"\n\n".join(
      f"[{i}][{'AI' if i % 2 == 0 else 'User'}] " + (
          "\n".join(
              item.get("text", "") if isinstance(item, dict) else str(item)
              for item in (m.get("content") if isinstance(m.get("content"), list) else [m.get("content",
  "")])
              if item is not None
          )
      )
      for i, m in enumerate(messages)
  )
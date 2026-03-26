# 部署LLM
git lfs install
git clone https://www.modelscope.cn/shawliu9/computerrl-glm4_1v-9b.git
git clone https://huggingface.co/Qwen/Qwen3.5-9B.git

conda activate uitars
export CUDA_VISIBLE_DEVICES=0
nohup python -m vllm.entrypoints.openai.api_server --served-model-name computerRL --model ~/projects/computerrl-glm4_1v-9b --gpu-memory-utilization 0.4 --port 30000 > autoglm.log 2>&1 &
nohup python -m vllm.entrypoints.openai.api_server --served-model-name uitars-1.5-7b --model ~/models/UI-TARS-1.5-7B --gpu-memory-utilization 0.4 --max-model-len 65536 --port 1235 --api-key xZj2JAV7rwdy5bgBia998eJXC5HiTWPiFxoQQ5tDDyg > uitars.log 2>&1 &

# 启动LLM
conda activate uitars
nohup python -m vllm.entrypoints.openai.api_server --served-model-name gta1-7b --model ~/models/GTA1-7B --gpu-memory-utilization 0.35 --max-model-len 65536 --host 0.0.0.0 --port 1234 --api-key xZj2JAV7rwdy5bgBia998eJXC5HiTWPiFxoQQ5tDDyg > gta1.log 2>&1 &
nohup python -m vllm.entrypoints.openai.api_server \
  --served-model-name qwen3.5-9b \
  --model ~/models/Qwen3.5-9B \
  --gpu-memory-utilization 0.4 \
  --max-model-len 65536 \
  --host 0.0.0.0 \
  --port 30000 \
  --api-key xZj2JAV7rwdy5bgBia998eJXC5HiTWPiFxoQQ5tDDyg > qwen.log 2>&1 &

# 启动qwen3.5-9b with reasoning and tool choice
nohup python -m vllm.entrypoints.openai.api_server \
  --served-model-name qwen3.5-9b \
  --model ~/models/Qwen3.5-9B \
  --gpu-memory-utilization 0.4 \
  --max-model-len 65536 \
  --host 0.0.0.0 \
  --port 30000 \
  --api-key xZj2JAV7rwdy5bgBia998eJXC5HiTWPiFxoQQ5tDDyg \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder > qwen.log 2>&1 &

# 启动embedding service
conda activate uitars
nohup python ~/projects/ComputerRL/mm_agents/autoglm_v_restart/embedding.py > embedding.log 2>&1 &

# 下载docker镜像
git lfs install
git clone https://huggingface.co/datasets/xlangai/ubuntu_osworld
cd ubuntu_osworld
unzip Ubuntu.qcow2.zip

配置/复制settings

# 加入docker用户组
sudo usermod -aG docker $USER
newgrp docker

# 更新server
sudo apt install -y qemu-utils
sudo ./update_server_main.sh

# 运行
# export OPENAI_BASE_URL="http://localhost:30000/v1"
# export OPENAI_API_KEY="EMPTY"
conda activate spider2v
nohup \
python run_autoglm_v.py \
    --provider_name docker \
    --path_to_vm ~/projects/ubuntu_osworld/Ubuntu.qcow2 \
    --result_dir results/computerRL_baseline \
    --headless \
    --max_steps 100 \
    --test_all_meta_path ./evaluation_examples/test_all.json \
    > nohup4.out 2>&1 &

nohup \
python run_autoglm_v_restart.py \
    --provider_name docker \
    --path_to_vm ~/projects/ubuntu_osworld/Ubuntu.qcow2 \
    --result_dir results/computerRL_gta1-7b_restart_ori_res \
    --visual_grounder_model gta1-7b \
    --screen_width 1920 \
    --screen_height 1080 \
    --headless \
    --max_steps 100 \
    --test_all_meta_path ./evaluation_examples/test_medium.json \
    > nohup3.out 2>&1 &

nohup \
python run_autoglm_v_restart.py \
    --provider_name docker \
    --path_to_vm ~/projects/ubuntu_osworld/Ubuntu.qcow2 \
    --result_dir results/computerRL_qwen3.5-9b_gta1-7b_restart \
    --model qwen3.5-9b \
    --temperature 0.1 \
    --top_p 0.95 \
    --repetition_penalty 1 \
    --presence_penalty 1.5 \
    --max_tokens 1024 \
    --visual_grounder_model gta1-7b \
    --headless \
    --max_steps 100 \
    --test_all_meta_path ./evaluation_examples/test_medium.json \
    --rerun_fail \
    > nohup2.out 2>&1 &

nohup \
python run_hisa.py \
  --provider_name docker \
  --path_to_vm ~/projects/ubuntu_osworld/Ubuntu.qcow2 \
  --result_dir ./results/hisa_qwen3.5-9b_wo_step_refinement_pattern_thinking \
  --headless \
  --max_steps 100 \
  --test_all_meta_path ./evaluation_examples/test_medium.json \
  --wo_step \
  --wo_refinement \
  --wo_pattern \
  --enable_thinking \
  > nohup.out 2>&1 &

nohup \
python run_hisa.py \
  --provider_name docker \
  --path_to_vm ~/projects/ubuntu_osworld/Ubuntu.qcow2 \
  --result_dir ./results/hisa_qwen3.5-9b_wo_refinement_pattern_thinking \
  --headless \
  --max_steps 100 \
  --test_all_meta_path ./evaluation_examples/test_medium.json \
  --wo_refinement \
  --wo_pattern \
  --enable_thinking \
  > nohup2.out 2>&1 &

nohup \
python run_hisa.py \
  --provider_name docker \
  --vm_ram 8G \
  --path_to_vm ~/projects/ubuntu_osworld/Ubuntu.qcow2 \
  --result_dir ./results/hisa_qwen3.5-9b_wo_pattern_thinking \
  --headless \
  --max_steps 100 \
  --test_all_meta_path ./evaluation_examples/test_all.json \
  --wo_pattern \
  --enable_thinking \
  > nohup3.out 2>&1 &

# 杀进程
ps -ef | grep run_autoglm_v_restart.py
pkill -f run_autoglm_v_restart.py
pkill -f run_hisa.py
ps -fp 2011314
tr '\0' ' ' < /proc/2010759/cmdline ; echo
fuser -k 8001/tcp

# debug
Image.open(BytesIO(base64.b64decode(messages[-1]['content'][-1]['image_url'].split(",")[1]))).save('tmp/tmp.jpg')

帮我分析一下~/projects/ComputerRL/results/hisa_qwen3.5-9b_wo_pattern_thinking下面的任务的
主要失败原因，只需要每个domain找两三个失败的例子分析一下，失败的例子汇总在~/projects/
ComputerRL/results/task_success_failures.xlsx的hisa_qwen3.5-9b_wo_pattern_thinking这一列为0的样
本，可以通过~/projects/ComputerRL/results/hisa_qwen3.5-9b_wo_pattern_thinking/*/*/
execution_log.json来看任务描述和每一步操作
通过改prompt(~/projects/ComputerRL/mm_agents/hisa/main.py GLOBAL_PLANNER_PROMPT)来优化

sudo npm install -g @openai/codex@latest

# openclaw
# 本地启动qwen3.5-9b
ssh -fN -L 30000:0.0.0.0:30000 CSE_T4
curl -v http://0.0.0.0:30000/v1/models

# 在openclaw中添加vLLM模型
openclaw onboard --install-daemon
    model provider: vLLM
        http://0.0.0.0:30000/v1
        xZj2JAV7rwdy5bgBia998eJXC5HiTWPiFxoQQ5tDDyg
openclaw configure
openclaw dashboard
python run_autoglm_v.py --provider_name vmware --path_to_vm D:\projects\OSWorld\vmware_vm_data\Ubuntu0\Ubuntu0.vmx --headless --max_steps 15 --test_all_meta_path ./evaluation_examples/test_one.json

python run_autoglm_v.py \
    --provider_name docker \
    --path_to_vm Ubuntu/Ubuntu.vmx \
    --headless \
    --max_steps 15 \
    --test_all_meta_path ./evaluation_examples/test_nogdrive.json
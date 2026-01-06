import logging
import os
import platform
import subprocess
import time
import traceback
from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.vmware.VMwareProvider")
logger.setLevel(logging.INFO)

WAIT_TIME = 3


def get_vmrun_type(return_list=False):
    if platform.system() == 'Windows' or platform.system() == 'Linux':
        if return_list:
            return ['-T', 'ws']
        else:
            return '-T ws'
    elif platform.system() == 'Darwin':  # Darwin is the system name for macOS
        if return_list:
            return ['-T', 'fusion']
        else:
            return '-T fusion'
    else:
        raise Exception("Unsupported operating system")


class VMwareProvider(Provider):
    @staticmethod
    def _execute_command(command: list, return_output=False, encoding='utf-8'):
        try:
            if return_output:
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding=encoding,
                    check=False
                )
                if result.returncode != 0:
                    logger.warning(f"Command {' '.join(command)} returned non-zero status: {result.returncode}")
                    logger.debug(f"Error output: {result.stderr}")
                return result.stdout.strip()
            else:
                result = subprocess.run(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding=encoding,
                    check=False
                )
                if result.returncode != 0:
                    logger.warning(f"Command {' '.join(command)} returned non-zero status: {result.returncode}")
                    logger.debug(f"Error output: {result.stderr}")
                return None
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            raise

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str):
        logger.info("Starting VMware VM...")

        max_retries = 5
        retry_count = 0
        retry_delay = 2  

        while True:
            try:
                
                output = subprocess.check_output(f"vmrun {get_vmrun_type()} list", shell=True, stderr=subprocess.STDOUT)
                output = output.decode('utf-8')
                output = output.splitlines()
                normalized_path_to_vm = os.path.abspath(os.path.normpath(path_to_vm))

                if any(os.path.abspath(os.path.normpath(line)) == normalized_path_to_vm for line in output):
                    logger.info("VM is running.")
                    break
                else:
                    _command = ["vmrun"] + get_vmrun_type(return_list=True) + ["start", path_to_vm]
                    if headless:
                        _command.append("nogui")
                    
                    # Use Popen without capturing output to avoid blocking in headless mode
                    # This matches the approach used in Spider2-V
                    logger.info(f"Starting VM with command: {' '.join(_command)}")
                    p = subprocess.Popen(_command)
                    p.wait()
                    
                    if p.returncode != 0:
                        # Command failed, but we can't get detailed error since we didn't capture output
                        # Try to determine if it's a lock issue by attempting again
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error(f"Failed to start VM after {max_retries} attempts")
                            raise Exception(f"Failed to start VM after {max_retries} attempts")
                        
                        logger.warning(f"VM start failed (attempt {retry_count}/{max_retries}), possibly due to lock")
                        logger.info(f"Attempting to clean lock files...")
                        VMwareProvider._try_clean_lock_files(path_to_vm)
                        logger.info(f"Waiting {retry_delay} seconds before retrying...")
                        time.sleep(retry_delay)
                        
                        retry_delay = min(retry_delay * 2, 30)
                        continue
                    
                    time.sleep(WAIT_TIME)

            except subprocess.CalledProcessError as e:
                error_output = e.output.decode('utf-8').strip() if hasattr(e, 'output') else str(e)
                logger.error(f"Error executing command: {error_output}")
                
                
                if ("failed to lock the file" in error_output.lower() or 
                    "another vmware process" in error_output.lower() or
                    "is in use by another process" in error_output.lower()):
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"Failed to acquire lock on VM after {max_retries} attempts")
                        raise Exception(f"Failed to start VM after multiple attempts: {error_output}")
                    
                    logger.warning(f"Lock acquisition failed (attempt {retry_count}/{max_retries})")
                    
                    logger.info("Attempting to clean lock files...")
                    VMwareProvider._try_clean_lock_files(path_to_vm)
                    time.sleep(retry_delay)
                    
                    retry_delay = min(retry_delay * 2, 30)
                else:
                    raise

    def get_ip_address(self, path_to_vm: str) -> str:
        logger.info("Getting VMware VM IP address...")
        retry_count = 0
        max_retries = 5
        
        while True:
            try:
                
                try:
                    vm_status_cmd = ["vmrun"] + get_vmrun_type(return_list=True) + ["list"]
                    vm_list_output = subprocess.check_output(vm_status_cmd, stderr=subprocess.STDOUT).decode('utf-8')
                    normalized_path_to_vm = os.path.abspath(os.path.normpath(path_to_vm))
                    
                    if normalized_path_to_vm not in vm_list_output:
                        if retry_count < max_retries:
                            logger.warning(f"VM is not powered on: {path_to_vm}. Attempting to start...")
                            self.start_emulator(path_to_vm, headless=True, os_type="Ubuntu")
                            retry_count += 1
                            time.sleep(WAIT_TIME * 2)  
                            continue
                        else:
                            raise Exception(f"Failed to start VM after {max_retries} attempts: {path_to_vm}")
                except Exception as e:
                    logger.error(f"Error checking VM status: {str(e)}")
                    if retry_count < max_retries:
                        retry_count += 1
                        time.sleep(WAIT_TIME)
                        continue
                    else:
                        raise Exception(f"Error checking VM status after {max_retries} attempts: {str(e)}")
                
                
                output = VMwareProvider._execute_command(
                    ["vmrun"] + get_vmrun_type(return_list=True) + ["getGuestIPAddress", path_to_vm, "-wait"],
                    return_output=True
                )
                logger.info(f"VMware VM IP address: {output}")
                return output
            except Exception as e:
                logger.error(f"Error getting IP address: {str(e)}")
                if retry_count < max_retries:
                    retry_count += 1
                    time.sleep(WAIT_TIME)
                    logger.info(f"Retrying to get VMware VM IP address... (Attempt {retry_count}/{max_retries})")
                else:
                    logger.error(f"Failed to get VM IP address after {max_retries} attempts")
                    raise

    def save_state(self, path_to_vm: str, snapshot_name: str):
        logger.info("Saving VMware VM state...")
        VMwareProvider._execute_command(
            ["vmrun"] + get_vmrun_type(return_list=True) + ["snapshot", path_to_vm, snapshot_name])
        time.sleep(WAIT_TIME)  # Wait for the VM to save

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        logger.info(f"Reverting VMware VM to snapshot: {snapshot_name}...")
        VMwareProvider._execute_command(
            ["vmrun"] + get_vmrun_type(return_list=True) + ["revertToSnapshot", path_to_vm, snapshot_name])
        time.sleep(WAIT_TIME)  # Wait for the VM to revert
        return path_to_vm

    def stop_emulator(self, path_to_vm: str):
        logger.info("Stopping VMware VM...")
        try:
            # Try to shutdown VM normally
            VMwareProvider._execute_command(["vmrun"] + get_vmrun_type(return_list=True) + ["stop", path_to_vm])
            time.sleep(WAIT_TIME)  # Wait for VM to stop

            # Check if the VM has actually stopped
            output = subprocess.check_output(f"vmrun {get_vmrun_type()} list", shell=True, stderr=subprocess.STDOUT)
            output = output.decode('utf-8')
            output = output.splitlines()
            normalized_path_to_vm = os.path.abspath(os.path.normpath(path_to_vm))

            # If VM is still running, try to force power off
            if any(os.path.abspath(os.path.normpath(line)) == normalized_path_to_vm for line in output):
                logger.warning("VM still running after stop command. Trying to power off...")
                VMwareProvider._execute_command(["vmrun"] + get_vmrun_type(return_list=True) + ["poweroff", path_to_vm, "hard"])
                time.sleep(WAIT_TIME)
            
            # Try to clean up lock files
            if not VMwareProvider._try_clean_lock_files(path_to_vm):
                logger.warning("Failed to clean lock files")
                
        except Exception as e:
            logger.error(f"Error stopping VM: {str(e)}")

    @staticmethod
    def _try_clean_lock_files(path_to_vm: str):
        try:
            vm_directory = os.path.dirname(path_to_vm)
            if not os.path.exists(vm_directory):
                logger.warning(f"VM directory does not exist: {vm_directory}")
                return False
                
            lock_files = [f for f in os.listdir(vm_directory) if f.endswith('.lck')]
            if not lock_files:
                logger.info("No lock files found to clean")
                return True
                
            logger.info(f"Found {len(lock_files)} lock files to clean")
            for lock_file in lock_files:
                lock_path = os.path.join(vm_directory, lock_file)
                if os.path.isdir(lock_path):
                    for root, dirs, files in os.walk(lock_path, topdown=False):
                        for name in files:
                            try:
                                os.remove(os.path.join(root, name))
                            except Exception as e:
                                logger.warning(f"Failed to remove lock file {name}: {str(e)}")
                        for name in dirs:
                            try:
                                os.rmdir(os.path.join(root, name))
                            except Exception as e:
                                logger.warning(f"Failed to remove lock directory {name}: {str(e)}")
                    try:
                        os.rmdir(lock_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove lock directory {lock_path}: {str(e)}")
                else:
                    try:
                        os.remove(lock_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove lock file {lock_path}: {str(e)}")
            
            logger.info("Lock files cleaned successfully")
            return True
        except Exception as e:
            logger.error(f"Error cleaning lock files: {str(e)}")
            return False

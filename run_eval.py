import multiprocessing
import os
import traceback
from framework.core import EvalConfig, Runner
from framework.registry import TaskRegistry
from framework.config import ConfigManager

import framework.evaluators.humaneval
import framework.evaluators.mbpp
import framework.evaluators.gsm8k

def main():
    try:
        # 1. Initialize ConfigManager
        config_manager = ConfigManager()
        
        # 2. Get task from config
        task_name = config_manager.get_global_setting("task")
        if not task_name:
            print("Error: No 'task' defined in config.yaml")
            return

        # 3. Get dataset path
        dataset_path = config_manager.get_dataset_path(task_name)
        if not dataset_path:
            print(f"Error: Dataset path for task '{task_name}' not defined in config.yaml")
            return
            
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset file not found: {dataset_path}")
            return

        # 4. Initialize EvalConfig (it loads from ConfigManager internally)
        print("Initializing configuration...")
        config = EvalConfig()
        
        print(f"Configuration Loaded:")
        print(f"  Task: {task_name}")
        print(f"  Model Profile: {config_manager.get_global_setting('selected_model')}")
        print(f"  Model Name: {config.model_name}")
        print(f"  Dataset: {dataset_path}")
        print(f"  Workers: {config.max_workers}")

        # 5. Get Task Class
        task_cls = TaskRegistry.get(task_name)
        if not task_cls:
            print(f"Error: Unknown task '{task_name}' registered in system.")
            print(f"Available tasks: {TaskRegistry.list_tasks()}")
            return
            
        # 6. Run
        print("Initializing task...")
        task = task_cls(dataset_path)
        print("Initializing runner...")
        runner = Runner(config)
        print("Starting execution...")
        runner.run(task, task_name)
        print("Execution finished.")

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

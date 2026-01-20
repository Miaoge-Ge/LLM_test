import json
import time
import re
from typing import List, Dict, Any
from ..core import BaseTask, LLMClient
from ..utils import CodeExecutor, format_time
from ..registry import TaskRegistry

@TaskRegistry.register("mbpp")
class MBPPTask(BaseTask):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.code_executor = CodeExecutor()
        self.header = "from typing import List, Dict, Tuple, Optional, Union, Any, Set, Deque\nimport math\nimport re\nimport sys\nimport heapq\nimport itertools\nimport collections\nimport functools\n"

    def load_data(self) -> List[Any]:
        problems = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        return problems

    @property
    def log_columns(self) -> List[str]:
        return ["task_id", "status", "duration", "tokens"]

    def process_item(self, item: Any, llm_client: LLMClient) -> Dict[str, Any]:
        task_id = str(item["task_id"])
        prompt = item["text"]
        test_list = item["test_list"]
        
        function_name_hint = ""
        if test_list and len(test_list) > 0:
            first_test = test_list[0]
            match = re.search(r"assert\s+(\w+)\(", first_test)
            if match:
                func_name = match.group(1)
                function_name_hint = f"\nImportant: The function name MUST be `{func_name}`."

        user_prompt = f"Task: {prompt}{function_name_hint}\n\nPlease write Python code to solve this task."
        system_prompt = "You are an expert Python programmer. Output only the code inside ```python``` blocks."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        start_time = time.time()
        completion, usage = llm_client.generate(messages, max_tokens=2048)
        duration = time.time() - start_time
        
        if not completion:
            return {
                "task_id": task_id,
                "status": "API_FAILED",
                "duration": format_time(duration),
                "duration_raw": duration,
                "tokens": 0
            }

        code = self._extract_code(completion)
        test_code_block = "\n".join(test_list)
        full_code = f"{self.header}\n{code}\n\n{test_code_block}"
        
        status, _ = self.code_executor.execute(full_code)
        
        return {
            "task_id": task_id,
            "status": status,
            "duration": format_time(duration),
            "duration_raw": duration,
            "tokens": usage.get("total_tokens", 0)
        }

    def _extract_code(self, text: str) -> str:
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        return text

import json
import time
import re
from typing import List, Dict, Any
from ..core import BaseTask, LLMClient
from ..utils import CodeExecutor, format_time
from ..registry import TaskRegistry

@TaskRegistry.register("humaneval")
@TaskRegistry.register("humanevalplus")
class HumanEvalTask(BaseTask):
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
        task_id = item["task_id"]
        prompt = item["prompt"]
        test_code = item["test"]
        entry_point = item["entry_point"]
        
        start_time = time.time()
        
        system_prompt = "You are an expert Python programmer. Complete the function based on the provided signature and docstring. Output only the code inside ```python``` blocks."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        completion, usage = llm_client.generate(messages)
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
        
        full_code = f"{self.header}\n{code}\n\n{test_code}\n\ncheck({entry_point})"
        
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
        if "def " in text:
            return text
        return text

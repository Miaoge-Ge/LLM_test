import time
import re
from typing import Dict, Any
from ..core import CodeGenerationTask, LLMClient
from ..utils import format_time
from ..registry import TaskRegistry

@TaskRegistry.register("humanevalplus")
class HumanEvalPlusTask(CodeGenerationTask):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.header = "from typing import List, Dict, Tuple, Optional, Union, Any, Set, Deque\nimport math\nimport re\nimport sys\nimport heapq\nimport itertools\nimport collections\nimport functools\nimport types\n\n_numpy = types.ModuleType('numpy')\n\ndef _is_seq(x):\n    return isinstance(x, (list, tuple))\n\ndef _allclose(a, b, rtol=1e-07, atol=0.0):\n    if _is_seq(a) and _is_seq(b):\n        if len(a) != len(b):\n            return False\n        return all(_allclose(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b))\n    try:\n        return abs(a - b) <= (atol + rtol * abs(b))\n    except Exception:\n        return a == b\n\n_numpy.allclose = _allclose\n_numpy.isclose = lambda a, b, rtol=1e-07, atol=0.0: _allclose(a, b, rtol=rtol, atol=atol)\n_numpy.ndarray = type('ndarray', (), {})\n_numpy.float64 = float\n_numpy.float32 = float\n_numpy.nan = float('nan')\n_numpy.inf = float('inf')\nsys.modules['numpy'] = _numpy\n"

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

        completion, usage, error_msg = llm_client.generate(messages)

        if error_msg:
            return {
                "task_id": task_id,
                "status": "CRITICAL_API_FAILURE",
                "error_msg": error_msg,
                "duration": format_time(time.time() - start_time),
                "duration_raw": time.time() - start_time,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }

        if not completion:
            return {
                "task_id": task_id,
                "status": "EMPTY_RESPONSE",
                "error_msg": "Empty completion (Possible Content Filter or Overload)",
                "duration": format_time(time.time() - start_time),
                "duration_raw": time.time() - start_time,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }

        code = self._extract_code(completion)
        if f"def {entry_point}" not in code:
            body = self._dedent_code(code)
            code = f"{prompt.rstrip()}\n{self._indent_code(body, 4)}"

        full_code = f"{self.header}\n{code}\n\n{test_code}\n\ncheck({entry_point})"

        return self._execute_and_log(task_id, full_code, start_time, usage)

    def _extract_code(self, text: str) -> str:
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
        elif "def " in text:
            code = text.strip()
        else:
            code = text.strip()

        return self._dedent_code(code)

    def _dedent_code(self, code: str) -> str:
        lines = code.split('\n')
        if len(lines) <= 1:
            return code.strip()

        min_indent = None
        for line in lines:
            stripped = line.lstrip()
            if stripped:  # Non-empty line
                indent = len(line) - len(stripped)
                min_indent = indent if min_indent is None else min(min_indent, indent)

        if not min_indent:
            return code

        dedented_lines = [(line[min_indent:] if line.strip() else "") for line in lines]

        return '\n'.join(dedented_lines)

    def _indent_code(self, code: str, spaces: int) -> str:
        prefix = " " * spaces
        return "\n".join((prefix + line if line.strip() else "") for line in code.split("\n"))

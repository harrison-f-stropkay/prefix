from __future__ import annotations

import sys
from pathlib import Path

from lm_eval.tasks import TaskManager

TASKS = ["piqa", "winogrande", "arc_easy", "hellaswag", "charbench"]
PROMPT_LIMIT = 10
NUM_FEWSHOT = 1


def _task_manager() -> TaskManager:
    tasks_dir = Path(__file__).resolve().parents[1] / "lm_eval_tasks"
    if not tasks_dir.exists():
        raise FileNotFoundError(f"lm_eval_tasks not found: {tasks_dir}")
    if str(tasks_dir) not in sys.path:
        sys.path.append(str(tasks_dir))
    return TaskManager(include_path=[str(tasks_dir)])


def _iter_docs(docs, limit: int):
    if hasattr(docs, "__len__"):
        return [docs[i] for i in range(min(limit, len(docs)))]
    out = []
    for doc in docs:
        out.append(doc)
        if len(out) >= limit:
            break
    return out


def main() -> None:
    task_manager = _task_manager()
    tasks = task_manager.load_task_or_group(TASKS)
    for name, task in tasks.items():
        task.set_fewshot_seed(0)
        docs = _iter_docs(task.eval_docs, PROMPT_LIMIT)
        target_delim = task.config.target_delimiter or " "
        print(f"\n=== {name} (showing {len(docs)}) ===")
        for idx, doc in enumerate(docs):
            context = task.fewshot_context(doc, NUM_FEWSHOT)
            choices = task.doc_to_choice(doc)
            gold = task.doc_to_target(doc)
            print(f"\n[{idx}] prompt:\n{context}")
            print("choices:")
            for c_idx, choice in enumerate(choices):
                full_input = f"{context}{target_delim}{choice}"
                print(f"  [{c_idx}] {choice!r}")
                print(f"      input: {full_input!r}")
            print(f"gold: {gold!r}")


if __name__ == "__main__":
    main()

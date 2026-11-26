"""Main entry point wrapper for running my-coding-agent directly via `python main.py`."""

from my_coding_agent.cli import (
    BASE_PROJECTS_DIR,
    coder_logger,
    handle_max_retries_exhausted,
    main,
    parse_plan_steps,
    planner_logger,
    run_agent_workflow,
    sys_logger,
    tester_logger,
)

__all__ = [
    "BASE_PROJECTS_DIR",
    "coder_logger",
    "handle_max_retries_exhausted",
    "main",
    "parse_plan_steps",
    "planner_logger",
    "run_agent_workflow",
    "sys_logger",
    "tester_logger",
]

if __name__ == "__main__":
    main()

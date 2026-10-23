# """Logger configuration module for file persistence, third-party noise suppression, and UI callbacks."""

# import logging
# from datetime import datetime
# from pathlib import Path
# from typing import Any, Dict

# import litellm
# from rich.console import Console
# from rich.markup import escape

# # Resolve project root directory dynamically (my-coding-agent root)
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# LOG_DIR = PROJECT_ROOT / "logs"

# # Global Rich console instance for formatted log callbacks
# console = Console()


# # def _custom_litellm_success_callback(
# #     kwargs: Dict[str, Any], completion_response: Any, start_time: Any, end_time: Any
# # ) -> None:
# #     """Render structured model invocation events directly to the terminal via Rich."""
# #     model = kwargs.get("model", "unknown-model")
    
# #     # Retrieve provider details from optional parameter maps or kwargs
# #     custom_llm_provider = kwargs.get("custom_llm_provider") or kwargs.get(
# #         "response_cost_calculator_optional_params", {}
# #     ).get("custom_llm_provider")

# #     provider_info = (
# #         f" [dim]via ({escape(str(custom_llm_provider))})[/dim]"
# #         if custom_llm_provider
# #         else ""
# #     )
    
# #     console.print(
# #         f"\n   [dim cyan]└─ 🌐 LiteLLM Request:[/] [bold magenta]{escape(str(model))}[/]{provider_info}"
# #     )
    


# def setup_file_logging() -> None:
#     """Initialize root file logger with date-based log files and configure UI log callbacks."""
#     LOG_DIR.mkdir(exist_ok=True)
#     today_str = datetime.now().strftime("%Y-%m-%d")
#     log_file_path = LOG_DIR / f"{today_str}.log"

#     # Configure root logger level
#     root_logger = logging.getLogger()
#     root_logger.setLevel(logging.INFO)

#     # Remove existing StreamHandlers to prevent raw console log leaking
#     for handler in root_logger.handlers[:]:
#         if isinstance(handler, logging.StreamHandler) and not isinstance(
#             handler, logging.FileHandler
#         ):
#             root_logger.removeHandler(handler)

#     # Attach file handler for persistent local logging
#     if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
#         file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
#         file_handler.setLevel(logging.INFO)
#         file_formatter = logging.Formatter(
#             "%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
#             datefmt="%Y-%m-%d %H:%M:%S",
#         )
#         file_handler.setFormatter(file_formatter)
#         root_logger.addHandler(file_handler)

#     # -------------------------------------------------------------------
#     # Third-Party Logger & LiteLLM Custom UI Interception
#     # -------------------------------------------------------------------
#     # Suppress internal debug information from LiteLLM stdout
#     litellm.suppress_debug_info = True

#     # Raise log levels for third-party modules to silence console output
#     for logger_name in ("LiteLLM", "litellm", "LiteLLM Router", "LiteLLM Proxy"):
#         logging.getLogger(logger_name).setLevel(logging.WARNING)

#     # Register custom callback for rich model invocation logging
#     if _custom_litellm_success_callback not in litellm.success_callback:
#         litellm.success_callback.append(_custom_litellm_success_callback)

"""Logger configuration module for file persistence and third-party noise suppression."""

import logging
from datetime import datetime
from pathlib import Path

import litellm

# Resolve project root directory dynamically (my-coding-agent root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"


def setup_file_logging() -> None:
    """Initialize root file logger with date-based log files."""
    LOG_DIR.mkdir(exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    log_file_path = LOG_DIR / f"{today_str}.log"

    # Configure root logger level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing StreamHandlers to prevent raw console log leaking
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            root_logger.removeHandler(handler)

    # Attach file handler for persistent local logging
    if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # -------------------------------------------------------------------
    # Third-Party Logger Noise Suppression
    # -------------------------------------------------------------------
    # Suppress internal debug information from LiteLLM stdout
    litellm.suppress_debug_info = True

    # Raise log levels for third-party modules to silence console output
    for logger_name in ("LiteLLM", "litellm", "LiteLLM Router", "LiteLLM Proxy"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
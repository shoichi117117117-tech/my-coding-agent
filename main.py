import sys
import re
from pathlib import Path
from config import Config
from agent import CodingAgent
from tools.obsidian import ObsidianAdapter
from tools.terminal import TerminalRunner
from tools.code_editor import CodeEditor

PROJECTS_DIR = "C:/Users/shoic/projects_with_agent"

def run_agent_workflow():
    """Execute the streamlined autonomous coding loop."""
    print("=" * 60)
    print("🚀 Gemini Autonomous Coding Agent (Streamlined)")
    print(f"📁 Obsidian Vault: {Config.OBSIDIAN_VAULT_PATH}")
    print(f"🔄 Max Retries: {Config.MAX_RETRIES}")
    print("=" * 60)

    try:
        agent = CodingAgent()
        obsidian = ObsidianAdapter(str(Config.OBSIDIAN_VAULT_PATH))
        terminal = TerminalRunner(default_timeout=30)
        # Select a folder where the agent works. 
        editor = CodeEditor(workspace_root=PROJECTS_DIR)
    except Exception as e:
        print(f"❌ Failed to initialize tools: {e}")
        sys.exit(1)

    # 1. User inputs ONLY the task instruction
    print("\n[Input Task]")
    task_instruction = input("👤 What do you want to build/fix?: ").strip()
    if not task_instruction:
        print("⚠️ Task instruction cannot be empty. Exiting.")
        return

    # 2. Gemini automatically determines the best target filename
    print("🧠 Inferred target file name using Gemini...")
    target_file = agent.suggest_target_file(task_instruction)
    target_file_path = Path(PROJECTS_DIR) / target_file
    print(f"📄 Target File Path: `{target_file_path}`")

    # 3. Automatically construct PowerShell test command
    # Append rule: Do not use interactive input()
    task_instruction += "\n\nIMPORTANT: Do NOT use interactive `input()`. Write unit tests or execution examples directly inside `if __name__ == '__main__':`."
    
    if target_file.endswith(".py"):
        test_command = f'python "{target_file_path}"'
    else:
        test_command = f"python -m pytest"  # Default fallback

    print(f"⚡ Auto Test Command (PowerShell): `{test_command}`")

    # Read Obsidian rules
    print("\n📖 Reading development rules from Obsidian...")
    obsidian_rules = obsidian.read_rules()

    execution_history = []
    retry_count = 0
    current_error = None

    print("\n" + "-" * 60)
    print(f"🎯 Executing Task: {task_instruction[:80]}...")
    print("-" * 60)

    # 4. Self-Correction Loop
    while True:
        existing_code = editor.read_file(target_file)

        print(f"\n🤖 [Gemini Agent] Generating solution (Attempt {retry_count + 1}/{Config.MAX_RETRIES})...")
        
        try:
            raw_response = agent.generate_code_solution(
                task_instruction=task_instruction,
                target_file_path=target_file,
                file_content=existing_code,
                obsidian_rules=obsidian_rules,
                error_context=current_error,
            )

            extracted_code = editor.extract_code_block(raw_response)
            editor.write_file(target_file, extracted_code)

        except Exception as api_err:
            print(f"❌ Error during code generation: {api_err}")
            obsidian.append_log(task_description=task_instruction, status="FAILED", error_log=str(api_err))
            break

        print(f"⚡ [PowerShell] Running: `{test_command}`...")
        exec_result = terminal.run(test_command)

        output_text = exec_result.stderr if not exec_result.success else exec_result.stdout
        execution_history.append({
            "attempt": retry_count + 1,
            "success": exec_result.success,
            "output": output_text
        })

        if exec_result.success:
            print("\n✅ [SUCCESS] Test execution passed cleanly!")
            if exec_result.stdout:
                print(f"Output:\n{exec_result.stdout[:200].strip()}")

            log_path = obsidian.append_log(
                task_description=task_instruction,
                status="SUCCESS",
                execution_history=execution_history,
            )
            print(f"📝 Log saved to Obsidian: `{log_path}`")
            break

        else:
            retry_count += 1
            current_error = exec_result.stderr or exec_result.stdout
            print(f"\n❌ [EXECUTION FAILED] Exit Code: {exec_result.returncode}")
            print(f"Error snippet:\n{current_error[:300].strip()}...")

            if retry_count >= Config.MAX_RETRIES:
                print("\n" + "=" * 60)
                print(f"⚠️ Reached maximum retry limit ({Config.MAX_RETRIES}). Handing off to human.")
                print("=" * 60)

                obsidian.append_log(
                    task_description=task_instruction,
                    status="FAILED (Max Retries Reached)",
                    execution_history=execution_history,
                )

                user_feedback = input("\n👤 Additional guidance (or 'q' to quit): ").strip()
                if user_feedback.lower() == "q" or not user_feedback:
                    print("👋 Task terminated.")
                    break

                task_instruction += f"\n\n[Human Feedback]: {user_feedback}"
                retry_count = 0
                current_error = None


if __name__ == "__main__":
    try:
        run_agent_workflow()
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled. Exiting.")
        sys.exit(0)
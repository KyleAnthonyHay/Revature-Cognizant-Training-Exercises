"""
LangGraph Studio Export File
============================
This file exports the agent for LangGraph Studio.
Run: langgraph dev
"""

import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool

load_dotenv()

model = init_chat_model("gpt-4o-mini", model_provider="openai")


@tool
def read_file(path: str) -> str:
    """Read contents of a file. Safe operation."""
    return f"[SIMULATED] Read file: {path}\nContent: This is simulated file content for {path}"


@tool
def list_directory(path: str) -> str:
    """List directory contents. Safe operation."""
    return f"[SIMULATED] Directory listing for {path}:\n- file1.txt\n- file2.log\n- subdirectory/"


@tool
def write_file(path: str, content: str) -> str:
    """Write to a file. MODERATE RISK - modifies filesystem."""
    return f"[SIMULATED] Successfully wrote {len(content)} characters to {path}"


@tool
def delete_file(path: str) -> str:
    """Delete a file. HIGH RISK - permanent data loss!"""
    return f"[SIMULATED] Successfully deleted file: {path}"


@tool
def execute_command(cmd: str) -> str:
    """Execute shell command. HIGH RISK - system access!"""
    return f"[SIMULATED] Executed command: {cmd}\nOutput: Command completed successfully"


hitl_middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        "read_file": False,
        "list_directory": False,
        "write_file": True,
        "delete_file": True,
        "execute_command": True,
    },
    description_prefix="⚠️ APPROVAL REQUIRED: This operation requires human approval before execution."
)


graph = create_agent(
    model=model,
    tools=[read_file, list_directory, write_file, delete_file, execute_command],
    middleware=[hitl_middleware],
    system_prompt="""You are a file management assistant with safety controls.

Available operations:
- SAFE (auto-approved): read_file, list_directory
- MODERATE RISK (requires approval): write_file
- HIGH RISK (requires approval): delete_file, execute_command

Always inform users about risk levels and explain what operations require approval.""",
    name="file_manager_agent"
)

"""
Pair Exercise 01: Approval System - Starter Code
=================================================
Build HITL approval for sensitive operations.

LEARNING GOALS:
- Import and configure HumanInTheLoopMiddleware
- Understand interrupt_on configuration
- Experience approval flow in LangSmith Studio
"""

import os
from dotenv import load_dotenv

from langchain.agents import create_agent
# TODO: Import HumanInTheLoopMiddleware
from langchain_core.tools import tool

load_dotenv()


# =============================================================================
# TODO 1: Import HITL Middleware
# =============================================================================
# Import HumanInTheLoopMiddleware from langchain.agents.middleware
# =============================================================================

# TODO: Add your import here


# =============================================================================
# TODO 2: Create Tools by Risk Level
# =============================================================================
# Create tools with different safety levels:
#
# Safe (no approval):
# - read_file: reads a file (simulated)
# - list_directory: lists directory contents (simulated)
#
# Moderate (requires approval):
# - write_file: writes to a file (simulated)
#
# Dangerous (definitely requires approval):
# - delete_file: deletes a file (simulated)
# - execute_command: runs a shell command (simulated)
#
# EXPERIMENT: What makes a good tool description for HITL?
# =============================================================================

@tool
def read_file(path: str) -> str:
    """Read contents of a file. Safe operation."""
    # TODO: Return simulated file contents
    pass


@tool
def list_directory(path: str) -> str:
    """List directory contents. Safe operation."""
    # TODO: Return simulated directory listing
    pass


@tool
def write_file(path: str, content: str) -> str:
    """Write to a file. MODERATE RISK - modifies filesystem."""
    # TODO: Return simulated write confirmation
    pass


@tool
def delete_file(path: str) -> str:
    """Delete a file. HIGH RISK - permanent data loss!"""
    # TODO: Return simulated deletion confirmation
    pass


@tool
def execute_command(cmd: str) -> str:
    """Execute shell command. HIGH RISK - system access!"""
    # TODO: Return simulated command output
    pass


# =============================================================================
# TODO 3: Configure HITL Middleware
# =============================================================================
# Create HumanInTheLoopMiddleware with:
# - interrupt_on: dict mapping tool names to True/False
#   - True = requires approval
#   - False = runs automatically
# - description_prefix: message shown in approval UI
#
# Configure so:
# - read_file, list_directory: no approval
# - write_file, delete_file, execute_command: require approval
#
# EXPERIMENT: What if all tools require approval?
# EXPERIMENT: Try different description_prefix messages
# =============================================================================

# TODO: Create the middleware
# hitl_middleware = HumanInTheLoopMiddleware(
#     interrupt_on={
#         "read_file": ???,
#         "list_directory": ???,
#         "write_file": ???,
#         "delete_file": ???,
#         "execute_command": ???,
#     },
#     description_prefix="???"
# )


# =============================================================================
# TODO 4: Create the Agent
# =============================================================================
# Create an agent that:
# - Uses all the file management tools
# - Uses your HITL middleware
# - Has a system prompt explaining risk levels
# - Does NOT include checkpointer (Studio handles it)
# =============================================================================

# TODO: Create the agent
# agent = create_agent(...)


# =============================================================================
# Testing - MUST USE LANGSMITH STUDIO
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("This exercise MUST be tested in LangSmith Studio!")
    print("=" * 60)
    print("\nRun: cd exercises && langgraph dev")
    print("\nThen in Studio, try:")
    print("- 'List files in /home' -> should run immediately")
    print("- 'Delete temp.log' -> should pause for approval")
    print("- 'Write hello to greeting.txt' -> should pause for approval")

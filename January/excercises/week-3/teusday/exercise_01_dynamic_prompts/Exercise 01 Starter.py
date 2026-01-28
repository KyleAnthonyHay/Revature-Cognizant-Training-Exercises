"""
Exercise 01: Dynamic Prompts - Starter Code
===========================================
Build an agent with context-aware system prompts.

LEARNING GOALS:
- Import and use @dynamic_prompt decorator
- Read from ModelRequest to access state
- Build adaptive prompts based on context
"""

import os
from datetime import datetime
from dotenv import load_dotenv

from langchain.agents import create_agent
# TODO: Import dynamic_prompt and ModelRequest
from langchain_core.tools import tool

load_dotenv()


# =============================================================================
# TODO 1: Import Dynamic Prompt Components
# =============================================================================
# Import from langchain.agents.middleware:
# - dynamic_prompt (decorator)
# - ModelRequest (type for the request object)
# =============================================================================

# TODO: Add your imports here
# from langchain.agents.middleware import ???, ???


# =============================================================================
# TODO 2: Create Tools
# =============================================================================

@tool
def explain_concept(topic: str) -> str:
    """Explain a programming concept."""
    # TODO: Return a helpful explanation
    pass


@tool
def write_code(description: str) -> str:
    """Write code for a given task description."""
    # TODO: Return sample code
    pass


# =============================================================================
# TODO 3: Implement Dynamic Prompt Middleware
# =============================================================================
# Create a function decorated with @dynamic_prompt that:
# 1. Receives a ModelRequest parameter
# 2. Reads experience_level from request.state (default: "intermediate")
# 3. Gets time of day from datetime.now().hour
# 4. Checks conversation length via len(request.messages)
# 5. Returns a customized system prompt string
#
# Adaptations to implement:
# - Beginner: simple language, analogies, step-by-step
# - Intermediate: balanced explanations with code
# - Expert: concise, technical terms freely
# - Morning: cheerful greeting
# - Long conversations: more concise responses
#
# EXPERIMENT: Add language preference adaptation
# EXPERIMENT: Add domain-specific adaptations (web dev vs data science)
# =============================================================================

# TODO: Implement the dynamic prompt function
# @dynamic_prompt
# def adaptive_prompt(request: ModelRequest) -> str:
#     # Get context from request
#     experience = request.state.get("experience_level", "intermediate")
#     hour = datetime.now().hour
#     message_count = len(request.messages)
#     
#     # TODO: Build and return your prompt
#     pass


# =============================================================================
# TODO 4: Create the Agent
# =============================================================================
# Create an agent that:
# - Uses your tools
# - Uses your adaptive_prompt middleware
# - Does NOT include checkpointer (for Studio export)
# =============================================================================

# TODO: Create the agent
# agent = create_agent(...)


# =============================================================================
# CLI Testing
# =============================================================================
if __name__ == "__main__":
    from langgraph.checkpoint.memory import InMemorySaver
    
    # TODO: Create a CLI version with checkpointer
    
    print("Test with different experience levels:")
    print("- Set 'experience_level': 'beginner' in state")
    print("- Set 'experience_level': 'expert' in state")
    print("- Ask the same question and compare responses!")

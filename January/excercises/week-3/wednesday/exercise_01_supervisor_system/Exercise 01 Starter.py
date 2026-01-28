"""
Exercise 01: Supervisor System - Starter Code
=============================================
Build a supervisor coordinating research and writing sub-agents.

LEARNING GOALS:
- Create sub-agents with create_agent
- Wrap sub-agents as tools
- Understand supervisor routing via tool calling
"""

import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()


# =============================================================================
# TODO 1: Create Sub-Agents
# =============================================================================
# Create two specialized agents:
#
# Research Agent:
# - Name: "research_agent"
# - System prompt explaining it gathers comprehensive information
# - Include CRITICAL note: must include ALL findings in response
#
# Writing Agent:
# - Name: "writing_agent"  
# - System prompt explaining it creates polished content
# - Include CRITICAL note: must include COMPLETE content in response
#
# Why the CRITICAL notes? Sub-agents only return their final message!
#
# EXPERIMENT: How does the system prompt affect output quality?
# EXPERIMENT: Try adding specific formatting instructions
# =============================================================================

# TODO: Create research_agent
# research_agent = create_agent(
#     name="???",
#     model="???",
#     tools=[],
#     system_prompt="???"
# )

# TODO: Create writing_agent
# writing_agent = create_agent(...)


# =============================================================================
# TODO 2: Create Tool Wrappers
# =============================================================================
# Wrap each sub-agent as a tool for the supervisor to call.
#
# research_topic tool:
# - Takes topic as parameter
# - Invokes research_agent with the topic
# - Extracts and returns the content from the result
#
# write_content tool:
# - Takes topic and research as parameters
# - Invokes writing_agent with both
# - Extracts and returns the content
#
# EXPERIMENT: What happens if you change the tool descriptions?
# EXPERIMENT: Add a "depth" parameter (brief vs comprehensive)
# =============================================================================

@tool
def research_topic(topic: str) -> str:
    """Research a topic thoroughly. Use this first to gather information."""
    # TODO: Invoke research_agent
    # result = research_agent.invoke({"messages": [{"role": "user", "content": ...}]})
    # 
    # TODO: Extract content from result
    # return result["messages"][-1].content
    pass


@tool
def write_content(topic: str, research: str) -> str:
    """Write polished content based on research. Use after researching."""
    # TODO: Invoke writing_agent with topic and research
    # TODO: Extract and return content
    pass


# =============================================================================
# TODO 3: Create Supervisor Agent
# =============================================================================
# Create the supervisor that:
# - Has both tools available
# - Has a system prompt explaining the workflow:
#   1. First research
#   2. Then write based on research
# - Presents final content to user
#
# EXPERIMENT: Does the supervisor always follow the right order?
# EXPERIMENT: What prompts cause it to skip research?
# =============================================================================

# TODO: Create the supervisor agent
# agent = create_agent(...)


# =============================================================================
# CLI Testing  
# =============================================================================
if __name__ == "__main__":
    print("Test prompts to try:")
    print("- 'Write a blog post about TypeScript benefits'")
    print("- 'Create an article about async/await in Python'")
    print("\nWatch LangSmith to see the tool call sequence!")

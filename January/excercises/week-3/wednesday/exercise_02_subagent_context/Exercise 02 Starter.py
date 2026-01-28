"""
Exercise 02: Sub-Agent Context - Starter Code
==============================================
Extend supervisor with context engineering.

LEARNING GOALS:
- Use ToolRuntime to access conversation context
- Use InjectedToolCallId for proper ToolMessage returns
- Use Command for state updates
"""

import os
from typing import Annotated
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
# TODO: Import additional required classes

load_dotenv()


# =============================================================================
# TODO 1: Import Required Classes
# =============================================================================
# You'll need:
# - ToolRuntime from langchain.tools
# - InjectedToolCallId from langchain.tools
# - ToolMessage from langchain_core.messages
# - Command from langgraph.types
# =============================================================================

# TODO: Add your imports here


# =============================================================================
# TODO 2: Create Sub-Agents
# =============================================================================
# Create research_agent and writing_agent similar to Exercise 1.
# These will be invoked by your context-aware tools.
# =============================================================================

# TODO: Create research_agent

# TODO: Create writing_agent


# =============================================================================
# TODO 3: Implement Research with Context
# =============================================================================
# Create a tool that:
# 1. Takes topic and runtime: ToolRuntime parameters
# 2. Gets conversation history from runtime.state.get("messages", [])
# 3. Formats recent context for the sub-agent
# 4. Invokes research_agent with the context
# 5. Returns the research result
#
# EXPERIMENT: How much context is helpful? Too much?
# EXPERIMENT: Does summarizing history vs raw messages matter?
# =============================================================================

@tool
def research_with_context(topic: str) -> str:
    """Research with access to conversation context.
    
    TODO: Add runtime: ToolRuntime parameter
    """
    # TODO: Get messages from runtime.state
    # messages = runtime.state.get("messages", [])
    
    # TODO: Format context (last N messages, summary, etc.)
    
    # TODO: Invoke research_agent with context
    
    # TODO: Return result
    pass


# =============================================================================
# TODO 4: Implement Writing with State Tracking
# =============================================================================
# Create a tool that:
# 1. Takes topic, research, tool_call_id (InjectedToolCallId), runtime
# 2. Invokes writing_agent
# 3. Gets existing topics_covered from runtime.state
# 4. Returns Command with:
#    - update containing ToolMessage and updated topics_covered list
#
# The Command allows you to both return a tool result AND update state!
#
# EXPERIMENT: What else could you track? Word count? Sentiment?
# =============================================================================

@tool
def write_with_tracking(topic: str, research: str) -> str:
    """Write content and track topics covered.
    
    TODO: Add tool_call_id: Annotated[str, InjectedToolCallId] parameter
    TODO: Add runtime: ToolRuntime parameter
    TODO: Change return type to Command
    """
    # TODO: Invoke writing_agent
    
    # TODO: Get existing topics from state
    # topics = runtime.state.get("topics_covered", [])
    
    # TODO: Return Command with ToolMessage and state update
    # return Command(update={
    #     "messages": [ToolMessage(content=..., tool_call_id=tool_call_id)],
    #     "topics_covered": topics + [topic]
    # })
    pass


# =============================================================================
# TODO 5: Create Supervisor Agent
# =============================================================================
# Create a supervisor that uses your context-aware tools.
# Include in prompt that it should track topics covered.
# =============================================================================

# TODO: Create the agent


# =============================================================================
# CLI Testing
# =============================================================================
if __name__ == "__main__":
    from langgraph.checkpoint.memory import InMemorySaver
    
    # TODO: Create CLI version with checkpointer
    
    print("Test with multiple related requests:")
    print("1. 'Write about Python basics'")
    print("2. 'Now write about advanced Python'")
    print("3. 'What topics have we covered?'")

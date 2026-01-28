"""
Exercise 02: Permission-Based Tools - Starter Code
===================================================
Build an agent with tier-based tool access.

LEARNING GOALS:
- Import and use @wrap_model_call decorator
- Filter tools based on state values
- Use request.override() to modify requests
"""

import os
from typing import Callable
from dotenv import load_dotenv

from langchain.agents import create_agent
# TODO: Import wrap_model_call, ModelRequest, ModelResponse
from langchain_core.tools import tool

load_dotenv()


# =============================================================================
# TODO 1: Import Model Call Wrapper Components
# =============================================================================
# Import from langchain.agents.middleware:
# - wrap_model_call (decorator)
# - ModelRequest (input type)
# - ModelResponse (output type)
# =============================================================================

# TODO: Add your imports here


# =============================================================================
# TODO 2: Create Tools by Tier
# =============================================================================
# Create tools for each subscription tier:
# - Free: basic operations accessible to all
# - Pro: advanced features for paid users
# - Enterprise: full capabilities
#
# EXPERIMENT: What tool descriptions help the agent recommend upgrades?
# =============================================================================

@tool
def run_basic_query(query: str) -> str:
    """Run a basic data query. Available to all users."""
    # TODO: Implement
    pass


@tool
def run_advanced_analysis(dataset: str, analysis_type: str) -> str:
    """Run advanced analysis. PRO tier and above only."""
    # TODO: Implement
    pass


@tool
def create_visualization(data: str, chart_type: str) -> str:
    """Create a visualization. PRO tier and above only."""
    # TODO: Implement
    pass


@tool
def export_data(format: str) -> str:
    """Export data to file. ENTERPRISE tier only."""
    # TODO: Implement
    pass


@tool
def sync_external(destination: str) -> str:
    """Sync to external system. ENTERPRISE tier only."""
    # TODO: Implement
    pass


# =============================================================================
# TODO 3: Define Tool Groups
# =============================================================================
# Create lists of tools for each tier:
# - FREE_TOOLS: basic only
# - PRO_TOOLS: basic + advanced features
# - ENTERPRISE_TOOLS: all tools
# =============================================================================

# TODO: Define your tool groups
FREE_TOOLS = []
PRO_TOOLS = []
ENTERPRISE_TOOLS = []


# =============================================================================
# TODO 4: Implement Permission Middleware
# =============================================================================
# Create a function decorated with @wrap_model_call that:
# 1. Receives request (ModelRequest) and handler (Callable)
# 2. Reads user_tier from request.state.get("user_tier", "free")
# 3. Selects the appropriate tool list based on tier
# 4. Uses request.override(tools=...) to modify available tools
# 5. Returns handler(modified_request)
#
# EXPERIMENT: What if you add a message explaining available features?
# EXPERIMENT: What happens if user asks for unavailable feature?
# =============================================================================

# TODO: Implement the permission middleware
# @wrap_model_call
# def permission_middleware(
#     request: ModelRequest,
#     handler: Callable[[ModelRequest], ModelResponse]
# ) -> ModelResponse:
#     # Get user tier
#     # Select tools
#     # Override request
#     # Return handler result
#     pass


# =============================================================================
# TODO 5: Create the Agent
# =============================================================================
# Create an agent with:
# - All tools (middleware will filter them)
# - Your permission middleware
# - A helpful system prompt that handles unavailable features gracefully
# =============================================================================

# TODO: Create the agent
# agent = create_agent(...)


# =============================================================================
# CLI Testing
# =============================================================================
if __name__ == "__main__":
    print("Test with different tiers in state:")
    print("- user_tier: 'free' -> only basic query available")
    print("- user_tier: 'pro' -> basic + advanced + viz")
    print("- user_tier: 'enterprise' -> all tools")
    print("\nTry: 'Create a visualization' as free user")

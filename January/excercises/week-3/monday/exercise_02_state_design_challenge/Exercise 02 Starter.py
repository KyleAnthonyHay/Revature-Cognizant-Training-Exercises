"""
Exercise 02: State Design Challenge - Starter Code
===================================================
Design and implement state with reducers for a support ticket workflow.

LEARNING GOALS:
- Use Annotated with custom reducers
- Design state for multi-step workflows
- Understand when to use reducers vs regular fields
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Literal, Annotated

# TODO: Import StateGraph, START, END from langgraph.graph  
# TODO: Import init_chat_model from langchain.chat_models

load_dotenv()


# =============================================================================
# TODO 1: Implement a Custom Reducer
# =============================================================================
# A reducer function receives (existing_value, new_value) and returns merged.
# Use for fields that should ACCUMULATE instead of REPLACE.
#
# Create log_reducer that:
# - Takes existing list and new list
# - Returns combined list (existing + new)
#
# EXPERIMENT: Add timestamps to log entries
# EXPERIMENT: Limit log to last N entries
# =============================================================================

def log_reducer(existing: list[str], new: list[str]) -> list[str]:
    """Reducer that appends new log entries to existing log."""
    # TODO: Implement
    pass


# =============================================================================
# TODO 2: Design the State with Reducers
# =============================================================================
# Create a TypedDict with:
# - ticket_content: str
# - customer_id: str
# - category: str (billing, technical, general)
# - urgency: int (1-5)
# - response_draft: str
# - processing_log: list[str] with your reducer
#
# Use Annotated[type, reducer] for fields that should accumulate.
#
# EXPERIMENT: Add a "tags" field that accumulates
# =============================================================================

class TicketState(TypedDict):
    """State for customer support ticket processing."""
    # TODO: Add your fields here
    # Regular field: field_name: type
    # Reducer field: field_name: Annotated[type, reducer_function]
    pass


# =============================================================================
# TODO 3: Implement Processing Nodes
# =============================================================================
# Each node should:
# - Process the ticket
# - Add an entry to processing_log (it will accumulate!)
# - Return only the fields being updated
#
# EXPERIMENT: What happens if you return processing_log without a list?
# =============================================================================

def categorize_ticket(state: TicketState) -> dict:
    """Categorize into billing, technical, or general."""
    # TODO: Use LLM to categorize
    # TODO: Return category AND log entry
    pass


def assess_urgency(state: TicketState) -> dict:
    """Assess urgency from 1 (low) to 5 (critical)."""
    # TODO: Use LLM to assess urgency
    # TODO: Return urgency AND log entry
    pass


def generate_response(state: TicketState) -> dict:
    """Generate response based on category and urgency."""
    # TODO: Use LLM considering category and urgency
    # TODO: Return response_draft AND log entry
    pass


def log_completion(state: TicketState) -> dict:
    """Add final log entry."""
    # TODO: Return log entry summarizing processing
    pass


# =============================================================================
# TODO 4: Build the Graph
# =============================================================================
# Linear flow: categorize -> assess -> generate -> log
#
# EXPERIMENT: Add conditional routing for urgent tickets
# =============================================================================

# TODO: Build and compile the graph


# =============================================================================
# Testing
# =============================================================================
if __name__ == "__main__":
    print("Test tickets after completing TODOs:")
    print("\nUrgent: 'Payment failed, charged twice! URGENT!'")
    print("Normal: 'How do I reset my password?'")
    print("Low: 'Thanks for great service!'")
    print("\nCheck that processing_log accumulates entries!")

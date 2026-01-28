"""
Exercise 01: First Workflow - Starter Code
==========================================
Build a document processing workflow with conditional routing.

LEARNING GOALS:
- Import and use StateGraph, START, END
- Add nodes and edges to build a workflow
- Implement conditional routing
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Literal

# TODO: Import StateGraph, START, END from langgraph.graph
# TODO: Import init_chat_model from langchain.chat_models

load_dotenv()


# =============================================================================
# TODO 1: Define the State
# =============================================================================
# Create a TypedDict with fields for:
# - document_text: str (the input document)
# - document_type: str (invoice, receipt, contract, unknown)
# - extracted_data: dict (extracted information)
# - formatted_output: str (final output)
#
# EXPERIMENT: Add a confidence_score field
# =============================================================================

class DocumentState(TypedDict):
    """State for document processing workflow."""
    # TODO: Add your fields here
    pass


# =============================================================================
# TODO 2: Implement Node Functions
# =============================================================================
# Each node function:
# - Takes state as parameter
# - Returns a dict with fields to update
# - Only returns the fields that change (not the whole state)
#
# classify_document:
# - Use LLM to classify as invoice/receipt/contract/unknown
# - Return {"document_type": "..."}
#
# extract_invoice_data, extract_receipt_data, extract_contract_data:
# - Use LLM to extract relevant fields
# - Return {"extracted_data": {...}}
#
# format_output:
# - Create a formatted string from the data
# - Return {"formatted_output": "..."}
#
# EXPERIMENT: What happens if a node returns nothing?
# =============================================================================

def classify_document(state: DocumentState) -> dict:
    """Classify the document type using an LLM."""
    # TODO: Create model with init_chat_model
    # TODO: Prompt the model to classify
    # TODO: Return the document_type
    pass


def extract_invoice_data(state: DocumentState) -> dict:
    """Extract invoice fields: amount, date, vendor."""
    # TODO: Implement
    pass


def extract_receipt_data(state: DocumentState) -> dict:
    """Extract receipt fields: total, store, items."""
    # TODO: Implement
    pass


def extract_contract_data(state: DocumentState) -> dict:
    """Extract contract fields: parties, terms, date."""
    # TODO: Implement
    pass


def format_output(state: DocumentState) -> dict:
    """Format the extracted data into readable output."""
    # TODO: Implement
    pass


# =============================================================================
# TODO 3: Implement Routing Function
# =============================================================================
# The routing function:
# - Takes state as parameter
# - Returns a STRING with the name of the next node
# - Called by add_conditional_edges()
#
# EXPERIMENT: What happens if you return an invalid node name?
# =============================================================================

def route_by_document_type(state: DocumentState) -> str:
    """Route to appropriate extraction node."""
    # TODO: Read document_type from state
    # TODO: Return the correct node name
    pass


# =============================================================================
# TODO 4: Build the Graph
# =============================================================================
# Steps:
# 1. Create StateGraph with your state class
# 2. Add nodes with add_node(name, function)
# 3. Add edge from START to classify
# 4. Add conditional edges from classify
# 5. Add edges from extraction nodes to format
# 6. Add edge from format to END
# 7. Compile the graph
#
# EXPERIMENT: Add a "validate" node between extraction and formatting
# =============================================================================

# TODO: Create the graph
# workflow = StateGraph(DocumentState)

# TODO: Add nodes
# workflow.add_node("classify", classify_document)
# ...

# TODO: Add edges
# workflow.add_edge(START, "classify")
# workflow.add_conditional_edges(
#     "classify",
#     route_by_document_type,
#     {
#         "invoice": "extract_invoice",
#         "receipt": "extract_receipt",
#         ...
#     }
# )

# TODO: Compile
# agent = workflow.compile()


# =============================================================================
# Testing
# =============================================================================
if __name__ == "__main__":
    print("Test documents after completing TODOs:")
    print("\nInvoice: 'Invoice #12345\\nAmount: $500'")
    print("Receipt: 'Store: Coffee Shop\\nTotal: $8'")
    print("Contract: 'Agreement between A and B dated Jan 1'")

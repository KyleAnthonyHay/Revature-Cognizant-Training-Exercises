"""
Session 1: Agent Memory Fundamentals - Demo Code

This demo covers:
- Part 1: Agent without memory (the problem)
- Part 2: Adding memory with InMemorySaver
- Part 3: Thread-based conversation isolation
- Part 4: Multi-user session patterns
- Part 5: Inspecting message history
- Part 6: State analysis and debugging

Prerequisites:
    pip install langchain langgraph langchain-openai langsmith

Usage:
    Comment/uncomment the function calls in main() to run specific demos.
"""

import os
from dotenv import load_dotenv

# =============================================================================
# API KEYS AND CONFIGURATION
# =============================================================================

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "session1-agent-memory")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")

# =============================================================================

from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================================
# PART 1: Agent WITHOUT Memory (The Problem)
# ============================================================================

def part1_agent_without_memory():
    """Demonstrates that agents forget without a checkpointer."""
    
    print("=" * 60)
    print("PART 1: Agent WITHOUT Memory")
    print("=" * 60)

    # Create agent without checkpointer - no memory!
    agent_no_memory = create_agent(
        model="openai:gpt-4o-mini",
        tools=[],
        system_prompt="You are a helpful assistant. Remember user details.",
        name="forgetful_agent"
    )

    # Turn 1: Share information
    result1 = agent_no_memory.invoke({
        "messages": [{"role": "user", "content": "Hi! My name is Alice and I love Python."}]
    })
    print(f"\nTurn 1 - User: Hi! My name is Alice and I love Python.")
    print(f"Turn 1 - Agent: {result1['messages'][-1].content}")

    # Turn 2: Agent forgets!
    result2 = agent_no_memory.invoke({
        "messages": [{"role": "user", "content": "What's my name and what do I love?"}]
    })
    print(f"\nTurn 2 - User: What's my name and what do I love?")
    print(f"Turn 2 - Agent: {result2['messages'][-1].content}")
    print("\n[!] Agent forgot - each call is independent without memory")


# ============================================================================
# PART 2: Agent WITH Memory (The Solution)
# ============================================================================

def part2_agent_with_memory():
    """Demonstrates memory with InMemorySaver checkpointer."""
    
    print("=" * 60)
    print("PART 2: Agent WITH Memory (InMemorySaver)")
    print("=" * 60)

    # Create checkpointer and agent with memory
    checkpointer = InMemorySaver()

    agent_with_memory = create_agent(
        model="openai:gpt-4o-mini",
        tools=[],
        system_prompt="You are a helpful assistant. Remember user details.",
        checkpointer=checkpointer,  # THIS ENABLES MEMORY
        name="memory_agent"
    )

    # Config with thread_id - required for memory to work
    config = {"configurable": {"thread_id": "alice_session_001"}}

    # Turn 1: Share information
    result1 = agent_with_memory.invoke(
        {"messages": [{"role": "user", "content": "Hi! My name is Alice and I love Python."}]},
        config  # Must pass config!
    )
    print(f"\nTurn 1 - User: Hi! My name is Alice and I love Python.")
    print(f"Turn 1 - Agent: {result1['messages'][-1].content}")

    # Turn 2: Agent remembers!
    result2 = agent_with_memory.invoke(
        {"messages": [{"role": "user", "content": "What's my name and what do I love?"}]},
        config  # Same config = same conversation
    )
    print(f"\nTurn 2 - User: What's my name and what do I love?")
    print(f"Turn 2 - Agent: {result2['messages'][-1].content}")
    print("\n[✓] Agent remembers! Memory works.")


# ============================================================================
# PART 3: Thread-Based Conversation Isolation
# ============================================================================

def part3_thread_isolation():
    """Demonstrates that different thread_ids have isolated memory."""
    
    print("=" * 60)
    print("PART 3: Thread Isolation (Multiple Users)")
    print("=" * 60)

    # Create fresh agent for this demo
    multi_user_agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[],
        system_prompt="You are a support agent. Remember each customer's name and issue.",
        checkpointer=InMemorySaver(),
        name="support_agent"
    )

    # Two separate users with different thread_ids
    config_alice = {"configurable": {"thread_id": "user_alice"}}
    config_bob = {"configurable": {"thread_id": "user_bob"}}

    # Alice's conversation
    multi_user_agent.invoke(
        {"messages": [{"role": "user", "content": "I'm Alice. I have a billing issue."}]},
        config_alice
    )
    print("\n[Alice] I'm Alice. I have a billing issue.")

    # Bob's conversation (completely separate)
    multi_user_agent.invoke(
        {"messages": [{"role": "user", "content": "I'm Bob. I need help with returns."}]},
        config_bob
    )
    print("[Bob] I'm Bob. I need help with returns.")

    # Alice asks about her issue - only sees her context
    result_alice = multi_user_agent.invoke(
        {"messages": [{"role": "user", "content": "What's my name and issue?"}]},
        config_alice
    )
    print(f"\n[Alice] What's my name and issue?")
    print(f"[Agent] {result_alice['messages'][-1].content}")

    # Bob asks about his issue - only sees his context
    result_bob = multi_user_agent.invoke(
        {"messages": [{"role": "user", "content": "What's my name and issue?"}]},
        config_bob
    )
    print(f"\n[Bob] What's my name and issue?")
    print(f"[Agent] {result_bob['messages'][-1].content}")

    print("\n[✓] Threads are completely isolated!")


# ============================================================================
# PART 4: Multi-Session Patterns
# ============================================================================

def part4_thread_naming_patterns():
    """Demonstrates different thread naming strategies."""
    
    print("=" * 60)
    print("PART 4: Thread Naming Patterns")
    print("=" * 60)

    def create_user_config(user_id: str) -> dict:
        """Pattern 1: One persistent conversation per user."""
        return {"configurable": {"thread_id": f"user_{user_id}"}}

    def create_session_config(user_id: str, session_id: str) -> dict:
        """Pattern 2: New conversation each session."""
        return {"configurable": {"thread_id": f"user_{user_id}_session_{session_id}"}}

    def create_topic_config(user_id: str, topic: str) -> dict:
        """Pattern 3: Separate conversation per topic."""
        return {"configurable": {"thread_id": f"user_{user_id}_topic_{topic}"}}

    # Demo: Same user, different topics
    topic_agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[],
        system_prompt="You are a helpful assistant.",
        checkpointer=InMemorySaver(),
        name="topic_agent"
    )

    # Orders topic
    orders_config = create_topic_config("charlie", "orders")
    topic_agent.invoke(
        {"messages": [{"role": "user", "content": "I want to discuss order #12345."}]},
        orders_config
    )
    print("\n[Orders Topic] User: I want to discuss order #12345.")

    # Support topic (separate conversation)
    support_config = create_topic_config("charlie", "support")
    topic_agent.invoke(
        {"messages": [{"role": "user", "content": "I need technical help with the API."}]},
        support_config
    )
    print("[Support Topic] User: I need technical help with the API.")

    # Return to orders - context preserved
    result = topic_agent.invoke(
        {"messages": [{"role": "user", "content": "What order were we discussing?"}]},
        orders_config
    )
    print(f"\n[Orders Topic] User: What order were we discussing?")
    print(f"[Agent] {result['messages'][-1].content}")


# ============================================================================
# PART 5: Inspecting Message History
# ============================================================================

def part5_message_history_inspection():
    """Demonstrates how to inspect message history and types."""
    
    print("=" * 60)
    print("PART 5: Message History Inspection")
    print("=" * 60)

    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        weather_data = {
            "austin": "Sunny, 85°F",
            "seattle": "Rainy, 55°F",
            "new york": "Cloudy, 68°F"
        }
        return weather_data.get(city.lower(), f"Weather unavailable for {city}")

    # Create agent with tools to see different message types
    inspection_agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[get_weather],
        system_prompt="You help with weather queries.",
        checkpointer=InMemorySaver(),
        name="weather_agent"
    )

    config = {"configurable": {"thread_id": "inspection_demo"}}

    # Build up some history
    inspection_agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in Austin?"}]},
        config
    )
    result = inspection_agent.invoke(
        {"messages": [{"role": "user", "content": "How about Seattle?"}]},
        config
    )

    # Inspect the message history
    print("\nMessage History:")
    print("-" * 40)
    for i, msg in enumerate(result['messages']):
        msg_type = type(msg).__name__
        content = str(msg.content)[:60] if msg.content else "(empty)"
        
        # Check for tool calls
        tool_info = ""
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tools = [tc.get('name', '?') for tc in msg.tool_calls]
            tool_info = f" [calls: {tools}]"
        
        print(f"  {i+1}. {msg_type:15}{tool_info}")
        print(f"      {content}")

    print(f"\nTotal messages: {len(result['messages'])}")


# ============================================================================
# PART 6: State Analysis and Debugging
# ============================================================================

def part6_state_analysis():
    """Demonstrates state analysis helper for debugging."""
    
    print("=" * 60)
    print("PART 6: State Analysis Helper")
    print("=" * 60)

    def analyze_conversation(result: dict) -> dict:
        """Analyze conversation state for debugging."""
        messages = result.get('messages', [])
        
        stats = {
            'total': len(messages),
            'human': 0,
            'ai': 0,
            'tool_calls': 0,
            'tool_results': 0
        }
        
        for msg in messages:
            msg_type = getattr(msg, 'type', type(msg).__name__).lower()
            
            if 'human' in msg_type:
                stats['human'] += 1
            elif 'ai' in msg_type:
                stats['ai'] += 1
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    stats['tool_calls'] += len(msg.tool_calls)
            elif 'tool' in msg_type:
                stats['tool_results'] += 1
        
        # Estimate tokens (rough: 4 chars per token)
        total_chars = sum(len(str(getattr(m, 'content', ''))) for m in messages)
        stats['estimated_tokens'] = total_chars // 4
        
        return stats

    @tool
    def lookup_product(product_id: str) -> str:
        """Look up product information."""
        products = {
            "P001": "Laptop Pro - $999",
            "P002": "Wireless Mouse - $29",
        }
        return products.get(product_id, f"Product {product_id} not found")

    # Create agent and build conversation
    debug_agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[lookup_product],
        system_prompt="You help with product inquiries.",
        checkpointer=InMemorySaver(),
        name="debug_agent"
    )

    config = {"configurable": {"thread_id": "debug_session"}}

    # Multiple turns
    debug_agent.invoke(
        {"messages": [{"role": "user", "content": "What's the price of P001?"}]},
        config
    )
    result = debug_agent.invoke(
        {"messages": [{"role": "user", "content": "And P002?"}]},
        config
    )

    # Use the analyzer
    stats = analyze_conversation(result)
    print("\nConversation Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


# ============================================================================
# MAIN - Comment/Uncomment to run specific demos
# ============================================================================

def main():
    """
    Main entry point.
    Comment/uncomment the function calls below to run specific demos.
    """
    
    # Part 1: Agent without memory (shows the problem)
    # part1_agent_without_memory()
    
    #Part 2: Agent with memory (the solution)
    # part2_agent_with_memory()
    
    # Part 3: Thread isolation for multiple users
    part3_thread_isolation()
    
    # Part 4: Thread naming patterns
    # part4_thread_naming_patterns()
    
    # Part 5: Message history inspection
    # part5_message_history_inspection()
    
    # Part 6: State analysis and debugging
    # part6_state_analysis()
    
    print("\n[!] Uncomment the demos you want to run in main()")


if __name__ == "__main__":
    main()

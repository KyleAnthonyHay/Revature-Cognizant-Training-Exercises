# Session 1: Agent Memory Fundamentals

## Overview

This session covers how LangChain agents remember information within and across conversations. You'll learn the memory architecture that makes agents feel intelligent and context-aware.

---

## 1. Why Memory Matters

By default, agents are **stateless** - each invocation is independent. Without memory:

```
Turn 1: User says "My name is Alice"
Turn 2: User asks "What's my name?" → Agent doesn't know!
```

Memory enables:
- Multi-turn conversations
- Context accumulation
- Personalized interactions

---

## 2. Two Types of Memory

LangChain distinguishes between two memory types:

| Aspect | State | Store |
|--------|-------|-------|
| **Scope** | Single conversation (thread) | Cross-conversation |
| **Lifetime** | Until thread ends | Permanent |
| **Management** | Automatic via checkpointer | Manual via tools |
| **Use Case** | Message history, session data | User preferences, learned facts |

**State** = "Remember what we discussed this session"  
**Store** = "Remember what we learned about this user forever"

---

## 3. Checkpointers: Enabling Memory

A **checkpointer** is the storage backend for agent state. It:
1. **Saves** state after each turn
2. **Loads** state on the next turn (using `thread_id`)

```
Turn 1                      Turn 2
┌─────────┐                ┌─────────┐
│ Invoke  │                │ Invoke  │
│ Agent   │                │ Agent   │
│         │                │         │
│ State   │──save──► [Checkpointer] ──load──►│ State   │
│ updated │                │ loaded  │
└─────────┘                └─────────┘
```

### InMemorySaver

The simplest checkpointer for development:

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    checkpointer=checkpointer,  # Enable memory
    name="memory_agent"
)
```

**Characteristics:**
- ✅ No setup required
- ✅ Fast (RAM access)
- ✅ Perfect for development
- ❌ Data lost on restart
- ❌ Not for production

### Checkpointer Options

| Use Case      | Checkpointer    | Persistence |
| ------------- | --------------- | ----------- |
| Development   | `InMemorySaver` | None (RAM)  |
| Local testing | `SqliteSaver`   | SQLite file |
| Production    | `PostgresSaver` | PostgreSQL  |

---

## 4. Thread-Based Conversations

The `thread_id` isolates conversations. Each thread has completely separate state.

```python
config = {
    "configurable": {
        "thread_id": "unique_conversation_id"
    }
}

result = agent.invoke({"messages": [...]}, config)
```

### Thread Isolation

```
Thread: "user_alice"          Thread: "user_bob"
├── "I love pizza"            ├── "I hate pizza"
├── Agent remembers           ├── Agent remembers
└── Alice's preference        └── Bob's preference

# These never mix!
```

### Thread Naming Patterns

| Pattern       | Example             | Use Case                      |
| ------------- | ------------------- | ----------------------------- |
| User-based    | `user_{user_id}`    | One conversation per user     |
| Session-based | `user_{id}_{uuid}`  | New conversation each session |
| Topic-based   | `user_{id}_{topic}` | Organized by subject          |

---

## 5. Message History

State primarily contains **messages** - the full conversation history.

### Message Types

| Type            | Role      | Purpose                |
| --------------- | --------- | ---------------------- |
| `SystemMessage` | system    | Instructions to agent  |
| `HumanMessage`  | user      | User input             |
| `AIMessage`     | assistant | Agent responses        |
| `ToolMessage`   | tool      | Tool execution results |

### Message Flow Example

```
1. SystemMessage: "You are a helpful assistant"
2. HumanMessage: "What's the weather?"
3. AIMessage: (calls weather tool)
4. ToolMessage: "72°F and sunny"
5. AIMessage: "The weather is 72°F and sunny"
```

### Message Accumulation

Messages grow with each turn:

```
Turn 1:  System + User + Assistant = 3 messages
Turn 5:  System + 10 messages = 11 messages
Turn 10: System + 20 messages = 21 messages
```

**Impact:**
- More tokens per API call
- Higher costs
- Potential context window limits

---

## 6. State vs Store: Decision Framework

Ask these questions:

1. **Will I need this in a future conversation?**
   - Yes → Store
   - No → State

2. **Is this part of the conversation flow?**
   - Yes → State (messages)
   - No → Consider Store

3. **Should this survive "clear history"?**
   - Yes → Store
   - No → State

### Examples

| Information          | Where to Store | Why                       |
| -------------------- | -------------- | ------------------------- |
| "I'm ordering pizza" | State          | Current conversation task |
| "I'm vegetarian"     | Store          | Permanent preference      |
| "Add mushrooms"      | State          | Part of current order     |
| "My timezone is PST" | Store          | Always relevant           |

---

## 7. Combining State and Store

Often you'll use both together:

```
┌─────────────────────────────────────────────────────────┐
│                        Agent                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   STATE (Checkpointer)         STORE (runtime.store)   │
│   ─────────────────────        ──────────────────────  │
│   • messages                   • user preferences      │
│   • current_task               • learned facts         │
│   • session_flags              • interaction history   │
│                                                        │
│   Automatic                    Manual tool operations  │
│   Thread-scoped                Namespace-scoped        │
│   Temporary                    Permanent               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 8. Debugging State

### Accessing Messages After Invocation

```python
result = agent.invoke({"messages": [...]}, config)

# View all messages
for msg in result["messages"]:
    print(type(msg).__name__, msg.content[:50])
```

### What to Check When Debugging

| Issue | Check |
|-------|-------|
| Agent doesn't remember | Is `thread_id` consistent? Is checkpointer configured? |
| Wrong tool called | Look at `tool_calls` in AIMessage |
| Context confusion | Are multiple threads mixed? |
| Slow responses | Count total messages (token usage) |

---

## 9. Key Patterns

### Pattern 1: Basic Memory Agent

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    checkpointer=InMemorySaver(),
    name="my_agent"
)

config = {"configurable": {"thread_id": "session_1"}}
agent.invoke({"messages": [{"role": "user", "content": "Hello"}]}, config)
```

### Pattern 2: Multi-User Sessions

```python
def get_config(user_id: str) -> dict:
    return {"configurable": {"thread_id": f"user_{user_id}"}}

# Each user gets isolated memory
agent.invoke({"messages": [...]}, get_config("alice"))
agent.invoke({"messages": [...]}, get_config("bob"))
```

### Pattern 3: Topic-Based Threads

```python
def get_topic_config(user_id: str, topic: str) -> dict:
    return {"configurable": {"thread_id": f"user_{user_id}_{topic}"}}

# Same user, separate conversations
agent.invoke({"messages": [...]}, get_topic_config("alice", "orders"))
agent.invoke({"messages": [...]}, get_topic_config("alice", "support"))
```

---

## 10. Common Mistakes to Avoid

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| No checkpointer | Agent forgets everything | Add `checkpointer=InMemorySaver()` |
| Forgetting config | Each call is isolated | Always pass config with `thread_id` |
| Same thread for all users | Users see each other's data | Include `user_id` in thread |
| Using deprecated patterns | Code breaks in v1.0 | Use `InMemorySaver`, not `ConversationBufferMemory` |

---

## Key Takeaways

1. **Agents are stateless by default** - checkpointers add memory
2. **InMemorySaver** is for development; use database-backed savers for production
3. **thread_id isolates conversations** - same ID = same memory
4. **State is session-scoped**, Store is permanent
5. **Messages accumulate** - plan for token growth
6. **Debug by inspecting messages** - check the full conversation history

---

## Additional Resources

- [LangGraph Persistence](https://docs.langchain.com/oss/python/langraph/concepts/persistence)
- [Memory Concepts](https://docs.langchain.com/oss/python/langchain/how-to/memory)
- [Message Types](https://docs.langchain.com/oss/python/langchain/concepts/messages)

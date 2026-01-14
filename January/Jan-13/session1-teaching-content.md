# Session 1: Foundation & Core Concepts
## LangChain v1.0 Training

---

## Session Overview

| Item | Details |
|------|---------|
| **Duration** | 2-2.5 hours |
| **Topics** | AWS Bedrock, LangChain v1.0 Architecture, Models, Invocation Patterns |

## Learning Objectives

By the end of this session, you will be able to:

1. Understand AWS Bedrock and its role in AI applications
2. Explain LangChain v1.0's architecture and philosophy
3. Identify deprecated v0.x patterns to avoid
4. Initialize models using `init_chat_model()` with provider strings
5. Apply the correct invocation pattern (invoke, batch, stream) for different scenarios

---

# 1. AWS Bedrock Orientation

## What is AWS Bedrock?

AWS Bedrock is a fully managed service that provides API access to foundation models from multiple providers. Think of it as a "model marketplace" where you access various AI models through a single, unified interface.

```
┌─────────────────────────────────────────────────────────────────┐
│                        AWS BEDROCK                              │
│                   "Model Marketplace"                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│   │ Anthropic│  │  Amazon  │  │   Meta   │  │ Mistral  │       │
│   │  Claude  │  │  Titan   │  │  Llama   │  │          │       │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                 │
│                     Unified API Access                          │
│                     Pay-per-use Pricing                         │
│                     Enterprise Security                         │
└─────────────────────────────────────────────────────────────────┘
```

## Key Benefits

| Benefit | Description |
|---------|-------------|
| **No infrastructure management** | AWS handles servers, scaling, and availability |
| **Multiple model providers** | Access Claude, Titan, Llama, and more from one API |
| **Pay-per-use pricing** | Only pay for tokens consumed |
| **Enterprise security** | Built-in IAM, VPC, and compliance integration |

## Available Model Providers

| Provider | Models | Best For |
|----------|--------|----------|
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Haiku | Complex reasoning, long context, agent tasks |
| **Amazon** | Titan Text, Titan Embeddings | General tasks, cost-effective embeddings |
| **Meta** | Llama 3.1, Llama 3.2 | Open-weight flexibility, fine-tuning |
| **Mistral** | Mistral Large, Mixtral | Efficient inference, multilingual support |

## IAM Permissions Required

To use Bedrock, your AWS IAM user or role needs these permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Action": [
            "bedrock:InvokeModel",
            "bedrock:InvokeModelWithResponseStream",
            "bedrock:ListFoundationModels"
        ],
        "Resource": "*"
    }]
}
```

## Environment Setup for Bedrock

```bash
# AWS credentials (in .env or environment)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

---

# 2. LangChain v1.0 Architecture

## The Core Philosophy: Simplicity First

LangChain v1.0 was redesigned with one principle: **"Simple things should be simple; complex things should be possible."**

```
v0.x: 50+ lines to create an agent
v1.0: 5 lines to create an agent
```

## The Four Building Blocks

LangChain v1.0 has four fundamental components:

```
┌──────────────────────────────────────────────────────────────┐
│                    LangChain v1.0                            │
├──────────────┬──────────────┬──────────────┬─────────────────┤
│    MODELS    │    TOOLS     │    AGENTS    │     MEMORY      │
│              │              │              │                 │
│ Chat models  │ @tool        │ create_      │ Checkpointers   │
│ LLM calls    │ decorator    │ agent()      │ State mgmt      │
│ Embeddings   │ Functions    │ Workflows    │ Thread IDs      │
└──────────────┴──────────────┴──────────────┴─────────────────┘
```

### 1. Models
The "brains" that power your applications. LangChain provides a unified interface:

```python
from langchain import init_chat_model

model = init_chat_model("openai:gpt-4o-mini")
model = init_chat_model("anthropic:claude-3-5-sonnet-20241022")
model = init_chat_model("bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0")
```

### 2. Tools
Functions that agents can call. The `@tool` decorator makes any Python function agent-callable:

```python
from langchain_core.tools import tool

@tool
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
```

### 3. Agents
Combine models with tools to create autonomous systems:

```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[calculate_sum],
    system_prompt="You are a helpful assistant.",
    name="math_agent"
)
```

### 4. Memory
Enables agents to maintain context across conversations:

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    checkpointer=InMemorySaver(),
    name="conversational_agent"
)
```

---

# 3. v0.x to v1.0 Migration

## Why Migration Knowledge Matters

Most online tutorials still show outdated v0.x patterns. Recognizing deprecated code prevents confusion and wasted debugging time.

## Critical Breaking Changes

### 1. LCEL (LangChain Expression Language) is Removed

**❌ v0.x - DEPRECATED:**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([...])
model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser  # LCEL pipe operator - REMOVED!
```

**✅ v1.0 - CORRECT:**
```python
from langchain import init_chat_model

model = init_chat_model("openai:gpt-4o-mini")
result = model.invoke([{"role": "user", "content": "Hello"}])
```

### 2. create_react_agent is Deprecated

**❌ v0.x - DEPRECATED:**
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
```

**✅ v1.0 - CORRECT:**
```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=tools,
    system_prompt="You are helpful.",
    name="my_agent"  # REQUIRED!
)
```

### 3. Memory Classes are Deprecated

**❌ v0.x - DEPRECATED:**
```python
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()
```

**✅ v1.0 - CORRECT:**
```python
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
```

## Migration Quick Reference

| v0.x Pattern | v1.0 Replacement |
|--------------|------------------|
| `prompt \| model \| parser` | `model.invoke()` |
| `create_react_agent()` | `create_agent()` |
| `AgentExecutor` | Not needed |
| `ConversationBufferMemory` | `InMemorySaver` checkpointer |
| `hub.pull("hwchase17/react")` | `system_prompt` parameter |
| `ChatOpenAI()` direct import | `init_chat_model("openai:...")` |

---

# 4. Introduction to Models (LLMs)

## What is a Large Language Model?

An LLM is a neural network trained to predict the next token in a sequence. Through this training, LLMs develop capabilities like:

- **Language understanding**: Parsing meaning from text
- **Language generation**: Producing coherent responses
- **Reasoning**: Following logical steps
- **In-context learning**: Adapting based on examples in the prompt

```
Input Tokens  →  [    LLM    ]  →  Output Tokens
"What is 2+2?"   [ Prediction ]     "The answer is 4."
```

## Chat Models vs. Completion Models

| Type | Input | Output | Status |
|------|-------|--------|--------|
| **Completion** | Raw text | Raw text | ❌ Deprecated |
| **Chat** | Messages with roles | Message | ✅ Use this |

**Always use chat models in LangChain v1.0.**

## Message Roles

Chat models work with structured messages:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "The capital is Paris."},
    {"role": "user", "content": "What's its population?"}
]
```

| Role | Purpose | Example |
|------|---------|---------|
| `system` | Set behavior and personality | "You are a pirate captain..." |
| `user` | User input | Questions, requests |
| `assistant` | Model's previous responses | For multi-turn context |
| `tool` | Tool execution results | "Weather: 72°F" |

## Model Selection Guide

```
                 Is speed critical?
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
           Yes                   No
            │                     │
    Use smaller/faster      Complex reasoning?
    (gpt-4o-mini, Haiku)          │
                         ┌────────┴────────┐
                         ▼                 ▼
                        Yes               No
                         │                 │
                   Use powerful       Use balanced
                   (Claude 3.5,      (gpt-4o-mini,
                    o1-mini)          Haiku)
```

**General Recommendations:**
- **Development/Testing**: GPT-4o-mini or Claude 3 Haiku (fast, cheap)
- **Production (standard)**: GPT-4o or Claude 3.5 Sonnet (balanced)
- **Complex reasoning**: Claude 3.5 Sonnet or o1-mini (most capable)

## Temperature Parameter

Temperature controls randomness in responses:

| Value | Behavior | Use Case |
|-------|----------|----------|
| `0.0` | Deterministic (same input = same output) | Factual tasks, code |
| `0.7` | Balanced creativity | General conversation |
| `1.0+` | High randomness | Brainstorming, creative writing |

---

# 5. The `init_chat_model()` Helper

## The Provider String Format

This is the standard syntax for specifying models in LangChain v1.0:

```
"provider:model-name"
```

## Provider String Examples

| Provider        | Format                    | Examples                                            |
| --------------- | ------------------------- | --------------------------------------------------- |
| **OpenAI**      | `openai:model_name`       | `openai:gpt-4o-mini`, `openai:gpt-4o`               |
| **Anthropic**   | `anthropic:model_name`    | `anthropic:claude-3-5-sonnet-20241022`              |
| **AWS Bedrock** | `bedrock:model_id`        | `bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0` |
| **Google**      | `google_genai:model_name` | `google_genai:gemini-1.5-pro`                       |

## Basic Usage

```python
from langchain import init_chat_model

# One line initialization
model = init_chat_model("openai:gpt-4o-mini")

# Invoke with messages
response = model.invoke([
    {"role": "user", "content": "Hello!"}
])
print(response.content)
```

## Configuration Options

```python
model = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0.0,     # 0.0 = deterministic, 1.0 = creative
    max_tokens=500,      # Limit response length
    timeout=30,          # Request timeout in seconds
    max_retries=2        # Retry count on failure
)
```

## Common Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `temperature` | float | Randomness (0.0-2.0) |
| `max_tokens` | int | Maximum output tokens |
| `timeout` | int | Request timeout in seconds |
| `max_retries` | int | Number of retries on failure |
| `api_key` | str | Override environment variable |

## Environment Variables

API keys are read automatically from environment variables:

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

---

# 6. Model Invocation Patterns

## The Three Core Patterns

| Pattern     | Use Case                      | Returns            |
| ----------- | ----------------------------- | ------------------ |
| `.invoke()` | Single synchronous request    | Complete response  |
| `.batch()`  | Multiple requests in parallel | List of responses  |
| `.stream()` | Real-time token output        | Iterator of chunks |

## Pattern 1: `.invoke()` - Single Requests

The most basic pattern. Send one request, wait for the complete response.

```python
from langchain import init_chat_model

model = init_chat_model("openai:gpt-4o-mini")

response = model.invoke([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
])

print(response.content)  # "The capital of France is Paris."
```

**When to use:**
- Simple Q&A interactions
- Single-turn operations
- When you need the complete response before proceeding

## Pattern 2: `.batch()` - Parallel Processing

Process multiple independent requests efficiently.

```python
questions = [
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
]

# Process all at once - much faster than sequential invoke()
responses = model.batch(questions)

for q, r in zip(questions, responses):
    print(f"Q: {q} → A: {r.content}")
```

**When to use:**
- Processing lists of items
- Generating multiple variations
- Any scenario with independent, parallelizable requests

**Performance comparison:**

| Approach | Time (5 questions) |
|----------|-------------------|
| 5x `.invoke()` sequentially | ~5-8 seconds |
| 1x `.batch()` | ~1-2 seconds |

**⚠️ Important:** Never use `.invoke()` in a loop when you can use `.batch()`!

## Pattern 3: `.stream()` - Real-Time Output

Get response tokens as they're generated, enabling responsive UIs.

```python
print("Response: ", end="")
for chunk in model.stream("Tell me a short story."):
    print(chunk.content, end="", flush=True)
print()
```

**When to use:**
- Chat interfaces where users see text appear in real-time
- Long-form generation where waiting feels too slow
- Any user-facing application where perceived speed matters

## Async Patterns

For async/await code patterns (FastAPI, etc.):

```python
import asyncio

async def main():
    model = init_chat_model("openai:gpt-4o-mini")
    
    # Async single request
    response = await model.ainvoke([
        {"role": "user", "content": "Hello!"}
    ])
    
    # Async batch
    responses = await model.abatch(["Q1", "Q2", "Q3"])

asyncio.run(main())
```

| Async Method | Sync Equivalent |
|--------------|-----------------|
| `.ainvoke()` | `.invoke()` |
| `.abatch()` | `.batch()` |
| `.astream()` | `.stream()` |

## Pattern Selection Decision Tree

```
                 How many requests?
                       │
            ┌──────────┴──────────┐
            │                     │
          Single              Multiple
            │                     │
    Need real-time?         Use .batch()
            │
     ┌──────┴──────┐
    Yes           No
     │             │
 .stream()    .invoke()


         Are you in async code?
                  │
         ┌───────┴────────┐
        Yes              No
         │                │
  Use .ainvoke()    Use .invoke()
  or .astream()     or .stream()
```

---

# Summary

## Key Takeaways

1. **AWS Bedrock** provides managed access to multiple AI providers through a unified API

2. **LangChain v1.0** follows "simplicity first" - agents can be created in 5 lines

3. **Avoid deprecated patterns:**
   - No LCEL (pipe operators)
   - No `create_react_agent()`
   - No `ConversationBufferMemory`

4. **`init_chat_model()`** is the standard for model initialization:
   ```python
   model = init_chat_model("provider:model-name")
   ```

5. **Invocation patterns:**
   - `.invoke()` for single requests
   - `.batch()` for multiple requests (always prefer over loops!)
   - `.stream()` for real-time UX

## Environment Setup Checklist

```bash
# Required packages
pip install langchain langchain-openai python-dotenv

# .env file
OPENAI_API_KEY=sk-your-key-here
```

## Quick Reference

```python
from langchain import init_chat_model

# Initialize
model = init_chat_model("openai:gpt-4o-mini", temperature=0)

# Single request
response = model.invoke("Hello!")

# Multiple requests
responses = model.batch(["Q1", "Q2", "Q3"])

# Streaming
for chunk in model.stream("Tell me a story"):
    print(chunk.content, end="")
```

---

## What's Next

**Session 2: Building Agents with Tools**
- Creating custom tools with `@tool` decorator
- Writing effective tool descriptions
- Using `create_agent()` function
- Testing tools independently

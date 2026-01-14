# Session 3: Observability & Debugging with LangSmith
## LangChain v1.0 Training

---

## Session Overview

| Item | Details |
|------|---------|
| **Duration** | 2-2.5 hours |
| **Topics** | LangSmith Setup, Tracing, Debugging, Cost Monitoring |

## Learning Objectives

By the end of this session, you will be able to:

1. Configure LangSmith for automatic tracing
2. Read and interpret trace hierarchies
3. Debug agent failures using the LangSmith dashboard
4. Monitor token usage and estimate costs
5. Identify performance bottlenecks in agent workflows

---

# 1. LangSmith Setup and Configuration

## What is LangSmith?

LangSmith is LangChain's observability and debugging platform. It automatically captures detailed information about every operation in your AI application.

```
┌─────────────────────────┐                 ┌─────────────────────────┐
│   Your Application      │                 │   LangSmith Dashboard   │
│                         │                 │                         │
│  ┌─────────────────┐    │    traces       │  ┌─────────────────┐    │
│  │  Agent Code     │────┼────────────────►│  │ Visual Timeline │    │
│  │                 │    │                 │  │ Token Counts    │    │
│  │  create_agent() │    │                 │  │ Cost Estimates  │    │
│  └─────────────────┘    │                 │  │ Debug Tools     │    │
│                         │                 │  └─────────────────┘    │
└─────────────────────────┘                 └─────────────────────────┘
```

## What LangSmith Captures

| Component | Captured Data |
|-----------|---------------|
| **Agent Runs** | Start time, end time, status, agent name |
| **LLM Calls** | Input prompts, output responses, token counts, latency |
| **Tool Calls** | Tool name, arguments, return value, execution time |
| **Errors** | Exception type, message, stack trace |
| **Costs** | Token usage and estimated costs |

## Environment Variables

LangSmith requires three environment variables:

```bash
# Required: Enable tracing
LANGSMITH_TRACING=true

# Required: Your API key from smith.langchain.com
LANGSMITH_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxx

# Optional: Project name (groups traces together)
LANGSMITH_PROJECT=my-project-name
```

## Getting Your API Key

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Sign up or log in
3. Click your profile icon → **Settings**
4. Navigate to **API Keys**
5. Click **Create API Key**
6. Copy the key immediately (you won't see it again)

## Complete .env File Example

```bash
# LangSmith Configuration
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxx
LANGSMITH_PROJECT=langchain-training

# LLM API Keys
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx
```

## Verifying Configuration

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Check required variables
required = ["LANGSMITH_TRACING", "LANGSMITH_API_KEY"]
for var in required:
    value = os.getenv(var)
    if value:
        display = value[:10] + "..." if "KEY" in var else value
        print(f"✓ {var} = {display}")
    else:
        print(f"✗ {var} = NOT SET")
```

## Project Organization

Use different projects to organize traces:

```python
import os

# Development
os.environ["LANGSMITH_PROJECT"] = "my-agent-dev"

# Staging
os.environ["LANGSMITH_PROJECT"] = "my-agent-staging"

# Production
os.environ["LANGSMITH_PROJECT"] = "my-agent-prod"
```

---

# 2. Tracing Agent Execution

## Automatic Tracing

Once LangSmith is configured, tracing is **automatic**. No code changes needed:

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def my_tool(query: str) -> str:
    """A simple tool."""
    return f"Result for {query}"

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[my_tool],
    name="demo_agent"
)

# This run is automatically traced
result = agent.invoke({
    "messages": [{"role": "user", "content": "Hello"}]
})
# Trace appears in LangSmith within seconds
```

## Trace Hierarchy

Traces form a tree structure showing parent-child relationships:

```
Run: "customer_support_agent" (4.2s total)
│
├── ChatOpenAI (0.8s)
│   ├── Input: [SystemMessage, HumanMessage]
│   └── Output: AIMessage with tool_call
│
├── search_knowledge_base (1.1s)
│   ├── Input: {"query": "return policy"}
│   └── Output: "Our return policy allows..."
│
├── ChatOpenAI (0.9s)
│   ├── Input: [SystemMessage, HumanMessage, AIMessage, ToolMessage]
│   └── Output: AIMessage (final response)
│
└── Total tokens: 847 (input: 623, output: 224)
```

## What Each Trace Node Shows

### LLM Calls
- Input messages (system, user, assistant, tool)
- Output message (response or tool call)
- Token counts (input/output)
- Model name and parameters
- Latency

### Tool Calls
- Function name
- Input arguments
- Return value
- Execution time

## Reading the Timeline

The timeline view shows execution order and duration:

```
Time →
├─────────────┬───────────┬─────────┬──────────────┤
│  ChatOpenAI │   tool:   │ChatOpenAI│   Response   │
│   (0.8s)    │  search   │ (0.5s)  │              │
│             │  (1.2s)   │         │              │
└─────────────┴───────────┴─────────┴──────────────┘
0s            0.8s        2.0s      2.5s
```

This immediately reveals:
- Total execution time (2.5s)
- Which component took longest (tool: 1.2s)
- Sequential vs. parallel execution

## Multi-Tool Traces

When an agent calls multiple tools:

```
Run: "research_agent"
│
├── ChatOpenAI (decides to call 2 tools)
│
├── search_web (0.8s)
│   └── "Found 5 relevant articles..."
│
├── search_database (0.6s)
│   └── "Found 3 matching records..."
│
├── ChatOpenAI (synthesizes results)
│
└── Final Response
```

---

# 3. Debugging with LangSmith Dashboard

## Dashboard Overview

The LangSmith dashboard has several key areas:

```
┌─────────────────────────────────────────────────────────────────┐
│ LangSmith                                            Profile ▼  │
├─────────────────────────────────────────────────────────────────┤
│ Projects    │ my-project                                        │
│             │ ┌─────────────────────────────────────────────────│
│ • Datasets  │ │ Recent Traces                                   │
│ • Prompts   │ │ ┌─────────────────────────────────────────────  │
│             │ │ │ ✓ support_agent    12:34  2.1s   Success     │
│             │ │ │ ✗ search_agent     12:31  0.4s   Error       │
│             │ │ │ ✓ qa_bot           12:28  1.8s   Success     │
│             │ │ └─────────────────────────────────────────────  │
│             │ └─────────────────────────────────────────────────│
└─────────────────────────────────────────────────────────────────┘
```

## Finding Traces

1. **Select your project** from the left sidebar
2. **View recent traces** in the main panel
3. **Filter traces** by:
   - Status (Success/Error)
   - Time range
   - Agent name
   - Search by content

## Inspecting a Trace

Click on any trace to see detailed information:

```
┌─────────────────────────────────────────────────────────────────┐
│ Run: customer_support_agent                                     │
│ Started: 12:34:56 PM | Duration: 2.1s | Tokens: 847            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Timeline                    Details                            │
│  ─────────                   ───────                            │
│  ▶ customer_support (2.1s)   Status: Success                   │
│    ├─ ChatOpenAI (0.8s)       Tokens: 847                       │
│    ├─ search_kb (1.0s)        Cost: $0.0012                    │
│    └─ ChatOpenAI (0.3s)                                        │
│                                                                 │
│  Click any node to see inputs/outputs                          │
└─────────────────────────────────────────────────────────────────┘
```

## Debugging Workflow

When something goes wrong, follow this process:

```
┌─────────────────────────────────────────────────────────────────┐
│                     DEBUGGING WORKFLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. FIND THE FAILED TRACE                                       │
│     → Look for red/error indicators in trace list               │
│     → Filter by status = "Error"                                │
│                                                                 │
│  2. IDENTIFY THE FAILURE POINT                                  │
│     → Expand the trace hierarchy                                │
│     → Find the step marked as failed                            │
│                                                                 │
│  3. EXAMINE THE INPUTS                                          │
│     → What was sent to the failing step?                        │
│     → Was the input malformed or unexpected?                    │
│                                                                 │
│  4. CHECK THE OUTPUT/ERROR                                      │
│     → Read the actual error message                             │
│     → What does it tell you about the root cause?               │
│                                                                 │
│  5. TRACE BACKWARDS                                             │
│     → If tool received bad input, check the LLM that called it  │
│     → Why did the LLM pass that input?                          │
│                                                                 │
│  6. FIX AND VERIFY                                              │
│     → Apply fix                                                 │
│     → Run same query again                                      │
│     → Compare new trace to old trace                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Common Issues and Solutions

| Issue | What to Look For | Solution |
|-------|------------------|----------|
| **Tool returns error** | ERROR in tool output | Add error handling in tool |
| **Wrong tool called** | LLM reasoning in trace | Improve tool descriptions |
| **Agent ignores tools** | No tool_calls in LLM output | Update system prompt |
| **Tool called with wrong params** | Check tool_call input | Fix parameter types/docs |
| **Infinite tool loop** | Same tool called repeatedly | Add exit condition |
| **Token limit exceeded** | Truncation in trace | Summarize or trim context |

## Comparing Traces

To understand differences between runs:

1. Select multiple traces using checkboxes
2. Click "Compare" button
3. View side-by-side execution

Use this for:
- Comparing successful vs. failed runs
- A/B testing prompt changes
- Debugging why similar inputs produced different outputs

---

# 4. Monitoring Token Usage and Costs

## How LLM Pricing Works

LLM providers charge per token (roughly 4 characters or 0.75 words):

| Provider/Model    | Input Cost (per 1M tokens) | Output Cost (per 1M tokens) |
| ----------------- | -------------------------- | --------------------------- |
| GPT-4o            | $2.50                      | $10.00                      |
| GPT-4o-mini       | $0.15                      | $0.60                       |
| Claude 3.5 Sonnet | $3.00                      | $15.00                      |
| Claude 3 Haiku    | $0.25                      | $1.25                       |

**Key insight:** Output tokens often cost more than input tokens, and smaller models are dramatically cheaper.

## Token Breakdown in Traces

Every trace shows token usage:

```
Run: my_agent (2.1s)
├── Total Tokens: 1,247
│   ├── Input: 892 tokens
│   └── Output: 355 tokens
├── Estimated Cost: $0.0156
└── Cost Breakdown:
    ├── LLM Call 1: 523 tokens ($0.0082)
    ├── LLM Call 2: 412 tokens ($0.0051)
    └── LLM Call 3: 312 tokens ($0.0023)
```

## What Consumes Tokens

| Component | Token Cost | Notes |
|-----------|------------|-------|
| System prompt | Input (each LLM call) | Repeated every call |
| User message | Input | Usually small |
| Conversation history | Input | Grows over time |
| Tool descriptions | Input (each call) | Adds up with many tools |
| Tool results | Input | Can be large |
| Model responses | Output (expensive) | Varies by task |

## Cost Optimization Strategies

### 1. Use Smaller Models When Appropriate

```python
# For simple tasks - use cheaper model
simple_agent = create_agent(
    model="openai:gpt-4o-mini",  # $0.15/$0.60 per 1M tokens
    tools=[...],
    name="simple_task_agent"
)

# For complex reasoning - use capable model
complex_agent = create_agent(
    model="openai:gpt-4o",  # $2.50/$10.00 per 1M tokens
    tools=[...],
    name="complex_task_agent"
)
```

### 2. Optimize System Prompts

```python
# ❌ Verbose prompt (many tokens every call)
system_prompt = """
You are an advanced AI assistant designed to help users with their questions.
You should always be polite, helpful, and informative. When answering questions,
try to provide comprehensive responses that fully address the user's needs.
If you don't know something, it's okay to say so. Remember to be concise but
thorough in your explanations...
"""  # ~80 tokens

# ✅ Concise prompt (fewer tokens)
system_prompt = """You are a helpful assistant. Be accurate and concise."""
# ~12 tokens
```

### 3. Optimize Tool Descriptions

```python
# ❌ Too verbose
@tool
def search(query: str) -> str:
    """
    This tool allows you to search our comprehensive database system
    for information about products, customers, orders, and more.
    The database contains millions of records spanning multiple years.
    You can search by any keyword and the tool will return relevant
    matches sorted by relevance score...
    """  # ~60 tokens

# ✅ Concise but clear
@tool
def search(query: str) -> str:
    """Search products/customers/orders. Returns top 5 matches."""
    # ~12 tokens
```

### 4. Limit Tool Response Size

```python
@tool
def search_documents(query: str) -> str:
    """Search documents and return summaries."""
    results = db.search(query)
    
    # Don't return full documents
    summaries = [r.summary[:200] for r in results[:5]]
    return "\n".join(summaries)
```

## Cost Estimation Formula

```python
def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate cost for a given model and token count."""
    pricing = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
    }
    
    rates = pricing.get(model, {"input": 0, "output": 0})
    
    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]
    
    return input_cost + output_cost

# Example
cost = estimate_cost(1000, 500, "gpt-4o-mini")
print(f"Estimated cost: ${cost:.6f}")  # $0.000450
```

---

# 5. Visualizing Agent Workflows

## Identifying Workflow Patterns

### Healthy Patterns

**Simple tool use:**
```
Agent
├── LLM (decide)
├── Tool (execute)
├── LLM (respond)
└── Done (3 steps, fast)
```

**Multi-tool workflow:**
```
Agent
├── LLM (analyze)
├── Tool A 
├── Tool B
├── LLM (synthesize)
└── Done (efficient)
```

### Warning Patterns

**Unnecessary loops:**
```
Agent
├── LLM → Tool A
├── LLM → Tool A (same thing!)
├── LLM → Tool A (again?)
└── Finally responds (wasted calls)
```

**Over-complicated flow:**
```
Agent
├── LLM → Tool A → LLM → Tool B → LLM → Tool C → LLM
└── Total: 7 steps for simple question
```

## Bottleneck Identification

In the timeline view, look for:

| Pattern | Indicates | Action |
|---------|-----------|--------|
| One step taking 80% of time | Bottleneck | Optimize that component |
| Sequential tools that could be parallel | Inefficiency | Consider parallel execution |
| Repeated tool calls | Possible loop | Check agent logic |
| Long LLM calls | Large context | Trim or summarize |

## Expected Patterns by Agent Type

| Agent Type | Expected Pattern | Typical Time |
|------------|------------------|--------------|
| **Simple Q&A** | LLM → Tool → LLM → Answer | 2-3 seconds |
| **Research** | LLM → Multiple Searches → LLM → Synthesis | 5-10 seconds |
| **Complex Task** | LLM → Planning → Multiple Tool Phases → Synthesis | 10+ seconds |

## Optimization Based on Traces

| Observation | Action |
|-------------|--------|
| Tool takes 70% of time | Optimize tool implementation or cache results |
| LLM called 5+ times | Consolidate into fewer calls |
| Same tool called repeatedly | Cache results or fix prompt |
| Long first LLM call | System prompt too complex |
| Many small tool calls | Batch into single tool |

---

# Summary

## Key Takeaways

1. **LangSmith Setup**
   - Three environment variables: `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`
   - Tracing is automatic once configured
   - Use projects to organize traces

2. **Reading Traces**
   - Traces show hierarchical execution
   - Each node has inputs, outputs, timing, tokens
   - Timeline reveals bottlenecks

3. **Debugging Workflow**
   - Find failed trace → Identify failure point → Examine inputs → Check error → Trace backwards → Fix and verify

4. **Cost Monitoring**
   - Track tokens per trace and project
   - Output tokens cost more than input
   - Optimize prompts, tools, and model selection

5. **Workflow Optimization**
   - Look for healthy vs. warning patterns
   - Identify bottlenecks in timeline
   - Compare traces to understand differences

## Quick Reference

```bash
# .env file
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_xxxxx
LANGSMITH_PROJECT=my-project
```

```python
# Tracing is automatic - just run your agent
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[my_tool],
    name="my_agent"
)

result = agent.invoke({"messages": [...]})
# Trace appears in LangSmith dashboard
```

## LangSmith Dashboard URL

**[smith.langchain.com](https://smith.langchain.com)**

---

## What's Next

This concludes the LangChain v1.0 Foundation Training:

- **Session 1:** Foundation & Core Concepts ✓
- **Session 2:** Building Agents with Tools ✓
- **Session 3:** Observability & Debugging ✓

Advanced topics for further learning:
- Memory and conversation persistence
- Multi-agent systems
- Production deployment patterns
- Advanced context engineering

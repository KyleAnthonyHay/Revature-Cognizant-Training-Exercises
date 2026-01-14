# Session 2: Building Agents with Tools
## LangChain v1.0 Training

---

## Session Overview

| Item | Details |
|------|---------|
| **Duration** | 2-2.5 hours |
| **Topics** | Tool Creation, Tool Routing, Agent Creation, Testing |

## Learning Objectives

By the end of this session, you will be able to:

1. Create custom tools using the `@tool` decorator
2. Write effective docstrings that enable proper agent routing
3. Build agents using `create_agent()` with proper configuration
4. Apply naming conventions for agents
5. Test tools independently before agent integration

---

# 1. Tool Creation with the `@tool` Decorator

## What is a Tool?

A tool is a Python function that an agent can call to interact with the outside world. Without tools, an agent can only generate text. With tools, an agent can:

- Search databases
- Call APIs
- Perform calculations
- Read/write files
- Execute any Python code

```
┌─────────────────────────────────────────────────────────────────┐
│                         AGENT                                   │
│                                                                 │
│   User Message → [LLM Reasoning] → Decision                     │
│                                        │                        │
│                         ┌──────────────┴──────────────┐         │
│                         ▼                             ▼         │
│                   Call a Tool               Respond Directly    │
│                         │                                       │
│                         ▼                                       │
│                   [Tool Executes]                               │
│                         │                                       │
│                         ▼                                       │
│                   Result → [LLM] → Final Response               │
└─────────────────────────────────────────────────────────────────┘
```

## The `@tool` Decorator

The `@tool` decorator from `langchain_core.tools` transforms any Python function into an agent-callable tool.

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Use when users ask about weather."""
    # Implementation here
    return f"The weather in {city} is sunny, 72°F."
```

## Anatomy of a Good Tool

A well-designed tool has four essential elements:

```python
from langchain_core.tools import tool

@tool
def calculate_tip(bill_amount: float, tip_percentage: float) -> str:
    """
    Calculate the tip for a restaurant bill.
    
    Use this tool when a user wants to know how much to tip.
    The tip_percentage should be a number like 15, 18, or 20 (not 0.15).
    
    Args:
        bill_amount: The total bill amount in dollars
        tip_percentage: The tip percentage (e.g., 15 for 15%)
    
    Returns:
        A string describing the tip amount and total
    """
    tip = bill_amount * (tip_percentage / 100)
    total = bill_amount + tip
    return f"Tip: ${tip:.2f}, Total: ${total:.2f}"
```

| Element                | Requirement                    | Why It Matters                         |
| ---------------------- | ------------------------------ | -------------------------------------- |
| **Type hints**         | All parameters must have types | Agent needs to know what to pass       |
| **Return type**        | Should return `str`            | Consistent format for agents           |
| **Docstring**          | Clear, descriptive             | Agent reads this to decide when to use |
| **Args documentation** | What each parameter means      | Helps agent pass correct values        |

## Parameter Types

Tools support various Python types:

### Basic Types

```python
@tool
def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}!"

@tool
def add_numbers(a: int, b: int) -> str:
    """Add two integers together."""
    return f"{a} + {b} = {a + b}"

@tool
def calculate_area(length: float, width: float) -> str:
    """Calculate the area of a rectangle."""
    return f"Area: {length * width} square units"
```

### Optional Parameters

```python
from typing import Optional

@tool
def search_products(
    query: str,
    category: Optional[str] = None,
    max_results: int = 5
) -> str:
    """
    Search for products in the catalog.
    
    Args:
        query: Search terms
        category: Optional category filter
        max_results: Maximum results to return (default: 5)
    """
    result = f"Searching for '{query}'"
    if category:
        result += f" in {category}"
    return f"{result}, returning up to {max_results} results"
```

### List Parameters

```python
from typing import List

@tool
def analyze_keywords(keywords: List[str]) -> str:
    """Analyze a list of keywords for trends."""
    return f"Analyzing {len(keywords)} keywords: {', '.join(keywords)}"
```

## Return Type Best Practice

**Always return strings** for best agent compatibility:

```python
# ✅ GOOD - String return
@tool
def get_count(category: str) -> str:
    """Count items in a category."""
    count = database.count(category)
    return f"There are {count} items in {category}."

# ❌ AVOID - Non-string returns can confuse agents
@tool
def get_count_bad(category: str) -> int:
    """Count items in a category."""
    return database.count(category)
```

## Tool Naming Conventions

The function name becomes the tool name. Follow these conventions:

```python
# ✅ Good names - snake_case, descriptive, verb-based
@tool
def search_products(query: str) -> str: ...

@tool
def send_email(to: str, subject: str) -> str: ...

@tool
def calculate_total(items: List[str]) -> str: ...

# ❌ Bad names - vague, unclear purpose
@tool
def do_thing(x: str) -> str: ...

@tool
def process(data: str) -> str: ...
```

---

# 2. Tool Descriptions and Routing

## How Agents Choose Tools

When an agent receives a message, it analyzes the user's request against all available tool descriptions to decide what action to take:

```
User Message
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  LLM analyzes message + ALL tool descriptions       │
│                                                     │
│  Tool 1: "Search products in catalog..."            │
│  Tool 2: "Check order status..."                    │
│  Tool 3: "Calculate shipping cost..."               │
└─────────────────────────────────────────────────────┘
     │
     ▼
Decision: Call Tool 2 (best match for "where is my order?")
```

**Critical Point:** The LLM only sees the docstring, not the implementation. Your docstring IS the tool's identity.

## What Makes a Good Description

A good tool description answers four questions:

1. **What does this tool do?** (Capability)
2. **When should I use it?** (Trigger conditions)
3. **When should I NOT use it?** (Exclusions)
4. **What should I pass to it?** (Parameter guidance)

```python
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the internal company knowledge base for information.
    
    USE THIS WHEN:
    - User asks about company policies
    - User needs internal documentation
    - User asks about procedures or guidelines
    
    DO NOT USE FOR:
    - General knowledge questions (use web search instead)
    - Questions about external companies
    - Current events or news
    
    Args:
        query: The search query describing what to find
    """
    return search_internal_docs(query)
```

## Common Description Mistakes

### Mistake 1: Too Vague

```python
# ❌ BAD - When should the agent use this?
@tool
def search(q: str) -> str:
    """Search for things."""
    ...

# ✅ GOOD - Clear purpose and trigger
@tool
def search_employee_directory(query: str) -> str:
    """
    Search the employee directory by name, department, or title.
    Use when users ask to find colleagues or look up contact information.
    """
    ...
```

### Mistake 2: Overlapping Descriptions

```python
# ❌ BAD - Agent can't distinguish between these
@tool
def tool_a(query: str) -> str:
    """Search for information."""
    ...

@tool
def tool_b(query: str) -> str:
    """Find information about topics."""
    ...

# ✅ GOOD - Clear differentiation
@tool
def search_public_web(query: str) -> str:
    """Search the public internet for general information and news."""
    ...

@tool
def search_internal_docs(query: str) -> str:
    """Search internal company documents and policies (not public internet)."""
    ...
```

### Mistake 3: Missing Parameter Guidance

```python
# ❌ BAD - What format should the date be?
@tool
def schedule_meeting(time: str) -> str:
    """Schedule a meeting."""
    ...

# ✅ GOOD - Clear format requirements
@tool
def schedule_meeting(time: str) -> str:
    """
    Schedule a calendar meeting.
    
    Args:
        time: Meeting datetime in ISO format (e.g., "2024-03-15T14:30:00")
    """
    ...
```

## Tool Differentiation Strategies

When you have similar tools, use **explicit differentiation**:

```python
@tool
def search_current_inventory(product_name: str) -> str:
    """
    Check CURRENT stock levels for a product.
    
    Use for: "Do we have X in stock?", "How many X are available?"
    Returns: Current quantity available for immediate purchase.
    
    NOT FOR: Historical sales data (use search_sales_history instead)
    """
    ...

@tool
def search_sales_history(product_name: str) -> str:
    """
    Look up HISTORICAL sales data for a product.
    
    Use for: "How did X sell last month?", "Sales trends for X?"
    Returns: Past sales figures and trends.
    
    NOT FOR: Current stock levels (use search_current_inventory instead)
    """
    ...
```

## Negative Constraints

Sometimes what a tool **shouldn't** be used for is equally important:

```python
@tool
def execute_sql(query: str) -> str:
    """
    Execute a read-only SQL query against the database.
    
    CAPABILITIES:
    - SELECT queries to retrieve data
    - Aggregate functions (COUNT, SUM, AVG)
    
    LIMITATIONS:
    - Cannot INSERT, UPDATE, or DELETE data
    - Cannot modify schema (CREATE, DROP, ALTER)
    - Maximum 1000 rows returned
    
    Args:
        query: A valid SELECT SQL query
    """
    ...
```

---

# 3. Agent Creation with `create_agent()`

## What is an Agent?

An agent is an LLM-powered system that can reason and take actions. Unlike simple chat models that only generate text, agents can:

1. **Observe** - Analyze the current situation
2. **Reason** - Decide what to do next
3. **Act** - Call tools or generate responses
4. **Repeat** - Continue until the task is complete

```
                    ┌─────────────────────────────┐
                    │         Agent               │
                    │                             │
User Message ──────►│  Observe → Reason → Act  ◄─────┐
                    │              │             │    │
                    │              ▼             │    │
                    │         Tool Call  ────────────►│
                    │              │             │    │
                    │              ▼             │
                    │      Generate Response    │
                    │              │             │
                    └──────────────┼─────────────┘
                                   ▼
                            Final Response
```

## The `create_agent()` Function

`create_agent()` is the LangChain v1.0 standard for building agents:

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

agent = create_agent(
    model="openai:gpt-4o-mini",           # Required: Model to use
    tools=[get_weather],                   # Required: List of tools
    system_prompt="You are a helpful assistant.",  # Optional: Instructions
    name="weather_agent"                   # Required: Agent name
)
```

## Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | str | Provider string (e.g., `"openai:gpt-4o-mini"`) |
| `tools` | List[Tool] | List of `@tool` decorated functions |
| `name` | str | Unique name for tracing and debugging |

## Optional Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `system_prompt` | str | Instructions that guide agent behavior |
| `checkpointer` | BaseCheckpointer | Memory storage (covered in Session 3) |

## Invoking Agents

Agents are invoked with a message dictionary:

```python
# Basic invocation
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What's the weather in Seattle?"}
    ]
})

# Access the response
response = result["messages"][-1].content
print(response)
```

## System Prompts Guide Behavior

The system prompt shapes how your agent behaves:

```python
# Minimal prompt
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[...],
    system_prompt="You are a helpful assistant.",
    name="basic_agent"
)

# Detailed prompt with guidance
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_orders, create_ticket],
    system_prompt="""You are a customer service agent for Acme Corp.

Your tools:
- search_orders: Look up order status
- create_ticket: Open support tickets

Guidelines:
- Always greet customers warmly
- Ask clarifying questions if the request is unclear
- Use search_orders before suggesting alternatives
- Escalate to a ticket if you can't resolve the issue

Tone: Professional but friendly.""",
    name="customer_service_agent"
)
```

## Multiple Tools Example

Agents can have multiple tools and decide which to use:

```python
@tool
def add(a: int, b: int) -> str:
    """Add two numbers. Use for addition or sum calculations."""
    return f"{a} + {b} = {a + b}"

@tool
def multiply(a: int, b: int) -> str:
    """Multiply two numbers. Use for multiplication or product calculations."""
    return f"{a} × {b} = {a * b}"

@tool
def divide(a: float, b: float) -> str:
    """Divide two numbers. Use for division or ratio calculations."""
    if b == 0:
        return "Error: Cannot divide by zero"
    return f"{a} ÷ {b} = {a / b}"

calculator = create_agent(
    model="openai:gpt-4o-mini",
    tools=[add, multiply, divide],
    system_prompt="You are a calculator. Use tools for all math operations.",
    name="calculator_agent"
)

# The agent chooses the right tool based on the question
result = calculator.invoke({
    "messages": [{"role": "user", "content": "What's 15 times 8?"}]
})
# Agent calls multiply(15, 8) → "15 × 8 = 120"
```

## Agents vs. Direct Model Calls

| Aspect               | Model (`.invoke()`) | Agent (`create_agent()`) |
| -------------------- | ------------------- | ------------------------ |
| Can call tools       | No                  | Yes                      |
| Multi-step reasoning | No                  | Yes                      |
| Automatic iteration  | No                  | Yes                      |
| Memory support       | No                  | Via checkpointer         |
| Best for             | Simple Q&A          | Complex tasks            |

---

# 4. Agent Naming Best Practices

## Why Names Are Required

In LangChain v1.0, every agent **must** have a name:

```python
# ❌ This will fail - no name provided
agent = create_agent(model="openai:gpt-4o-mini", tools=[])

# ✅ This works
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    name="my_agent"  # Required!
)
```

## Why Names Matter

| Purpose | Benefit |
|---------|---------|
| **Debugging** | Identify which agent produced which output |
| **LangSmith Tracing** | Filter and search traces by agent name |
| **Multi-Agent Systems** | Know which agent performed each action |
| **Logging** | Clear audit trails in production |

## Naming Conventions

### Use snake_case

```python
# ✅ Good - snake_case
name="weather_agent"
name="customer_service_bot"
name="document_search_agent"

# ❌ Avoid - other cases
name="WeatherAgent"        # PascalCase
name="weather-agent"       # kebab-case
name="WEATHER_AGENT"       # SCREAMING_CASE
```

### Be Descriptive

```python
# ❌ Too generic - doesn't tell you what it does
name="agent"
name="bot"
name="my_agent"
name="test"

# ✅ Descriptive - purpose is clear
name="product_recommendation_agent"
name="order_status_checker"
name="technical_support_assistant"
name="meeting_scheduler_bot"
```

### Common Naming Patterns

| Pattern              | Example             | Use Case              |
| -------------------- | ------------------- | --------------------- |
| `{function}_agent`   | `search_agent`      | Single-purpose agents |
| `{domain}_assistant` | `finance_assistant` | Domain specialists    |
| `{task}_bot`         | `scheduler_bot`     | Task-specific bots    |
| `{team}_{role}`      | `sales_researcher`  | Multi-agent teams     |

## Names in LangSmith

When viewing traces in LangSmith, agent names appear in the execution graph:

```
Run: "product_recommendation_agent"
├── ChatOpenAI
├── ToolCall: search_products
│   └── search_products executed
├── ChatOpenAI
└── Response: "Based on your preferences..."
```

Good names make traces immediately understandable.

## What Names to Avoid

```python
# ❌ Too generic
name="agent1"
name="test"
name="main"

# ❌ Too long
name="the_agent_that_handles_customer_service_inquiries_for_premium_users"

# ❌ Special characters (may cause issues)
name="agent@v2"
name="agent.backup"
name="agent#1"

# ❌ Confusing abbreviations
name="csa"   # Customer service agent? Content search?
name="pra1"  # What does this mean?
```

---

# 5. Testing Tools Independently

## Why Test Tools Before Integration?

When your agent isn't working correctly, is the problem in the tool or the agent's reasoning? If you haven't tested your tools independently, you can't answer this question.

**The Testing Hierarchy:**

```
Level 1: Direct Function Call
    ↓
Level 2: Tool Invocation (via .invoke())
    ↓
Level 3: Integration with Agent
```

Problems at Level 1 will cascade up. **Test from the bottom.**

## Level 1: Test the Raw Function

Tools are just Python functions. Test them as functions first:

```python
@tool
def calculate_discount(price: float, discount_percent: float) -> str:
    """Calculate the discounted price."""
    if discount_percent < 0 or discount_percent > 100:
        return "Error: Discount must be between 0 and 100"
    
    final_price = price * (1 - discount_percent / 100)
    return f"Original: ${price:.2f}, Final: ${final_price:.2f}"

# Test the underlying function logic
print(calculate_discount.func(100, 20))
# "Original: $100.00, Final: $80.00"

print(calculate_discount.func(100, -5))
# "Error: Discount must be between 0 and 100"
```

## Level 2: Test the Tool Interface

Use `.invoke()` to test the tool as LangChain sees it:

```python
# Test via the tool interface
result = calculate_discount.invoke({
    "price": 100,
    "discount_percent": 20
})
print(result)
# "Original: $100.00, Final: $80.00"

# Test with invalid input
result = calculate_discount.invoke({
    "price": 100,
    "discount_percent": 150
})
print(result)
# "Error: Discount must be between 0 and 100"
```

## Common Tool Errors

### Error 1: Missing Type Hints

```python
# ❌ BAD - No type hints
@tool
def bad_tool(x, y):
    """Add two numbers."""
    return x + y
# Error: Schema cannot be inferred

# ✅ GOOD - Proper type hints
@tool
def good_tool(x: int, y: int) -> str:
    """Add two numbers."""
    return f"{x} + {y} = {x + y}"
```

### Error 2: Non-String Returns

```python
# ❌ BAD - Returns dict (can confuse agents)
@tool
def get_user(user_id: str) -> dict:
    """Get user details."""
    return {"name": "Alice", "age": 30}

# ✅ GOOD - Returns string
@tool
def get_user(user_id: str) -> str:
    """Get user details."""
    return "User: Alice, Age: 30"
```

### Error 3: Unhandled Exceptions

```python
# ❌ BAD - Exception crashes the agent
@tool
def risky_divide(a: float, b: float) -> str:
    """Divide two numbers."""
    return str(a / b)  # Crashes on b=0!

# ✅ GOOD - Graceful error handling
@tool
def safe_divide(a: float, b: float) -> str:
    """Divide two numbers."""
    if b == 0:
        return "Error: Cannot divide by zero"
    return f"{a} ÷ {b} = {a / b}"
```

### Error 4: Missing or Poor Docstrings

```python
# ❌ BAD - No docstring
@tool
def mystery(x: str) -> str:
    return x.upper()

# ✅ GOOD - Clear docstring
@tool
def convert_to_uppercase(text: str) -> str:
    """
    Convert text to uppercase.
    Use when the user wants text in all capital letters.
    """
    return text.upper()
```

## Edge Case Testing

Test your tools with edge cases:

```python
@tool
def search_products(query: str, max_price: float = None) -> str:
    """Search for products, optionally filtering by max price."""
    if not query.strip():
        return "Error: Query cannot be empty"
    if max_price is not None and max_price < 0:
        return "Error: Max price cannot be negative"
    return f"Found products matching '{query}'"

# Test cases
print(search_products.invoke({"query": "laptop"}))           # Normal
print(search_products.invoke({"query": ""}))                 # Empty query
print(search_products.invoke({"query": "laptop", "max_price": -50}))  # Negative price
print(search_products.invoke({"query": "laptop", "max_price": 0}))    # Zero price
```

## Tool Testing Checklist

Before integrating a tool with an agent, verify:

| Check | Test |
|-------|------|
| ✅ Normal inputs work | `tool.invoke({"param": valid_value})` |
| ✅ Edge cases handled | Zero, negative, empty, max values |
| ✅ Invalid inputs return errors | Don't crash, return helpful message |
| ✅ Return values are clear | Strings with useful information |
| ✅ Docstring is accurate | Describes when to use the tool |

## Inspecting Tool Metadata

Check what the agent sees:

```python
@tool
def my_tool(query: str) -> str:
    """Search for information."""
    return "results"

# Inspect tool metadata
print(f"Name: {my_tool.name}")
print(f"Description: {my_tool.description}")
print(f"Args Schema: {my_tool.args_schema.schema()}")
```

---

# Summary

## Key Takeaways

1. **`@tool` decorator** transforms functions into agent-callable tools
   - Always use type hints
   - Always return strings
   - Docstrings are critical

2. **Tool descriptions** determine when agents use tools
   - Answer: What, When, When NOT, How
   - Differentiate similar tools explicitly
   - Include negative constraints

3. **`create_agent()`** is the v1.0 standard
   - Required: model, tools, name
   - Optional: system_prompt, checkpointer
   - Invoke with `{"messages": [...]}`

4. **Agent names are required**
   - Use snake_case
   - Be descriptive
   - Essential for debugging and tracing

5. **Test tools independently**
   - Level 1: Direct function call
   - Level 2: Tool `.invoke()`
   - Level 3: Agent integration

## Quick Reference

```python
from langchain_core.tools import tool
from langchain.agents import create_agent

# Create a tool
@tool
def my_tool(param: str) -> str:
    """Clear description of what this does and when to use it."""
    return f"Result for {param}"

# Test the tool
result = my_tool.invoke({"param": "test"})

# Create an agent
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[my_tool],
    system_prompt="You are helpful.",
    name="my_agent"
)

# Invoke the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "Hello"}]
})
response = result["messages"][-1].content
```

---

## What's Next

**Session 3: Observability & Debugging with LangSmith**
- Setting up LangSmith tracing
- Exploring execution traces
- Debugging agent failures
- Monitoring token usage and costs

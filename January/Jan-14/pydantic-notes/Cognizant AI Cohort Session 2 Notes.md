# Session 2: Structured Output & Validation

## Overview

This session covers how to get reliable, typed responses from LLMs using Pydantic schemas. You'll learn to define output structures, validate responses, and handle errors gracefully.

---

## 1. The Problem with Unstructured Output

By default, LLMs return free-form text:

```
User: "Analyze this review: Great product, fast shipping, bad packaging"

LLM: "This review is mostly positive. The customer liked the product 
      and shipping speed but had concerns about packaging..."
```

**Problems:**
- How do you extract the sentiment programmatically?
- How do you get a numeric rating?
- How do you ensure consistent format every time?

**Solution:** Structured output with Pydantic schemas.

---

## 2. Pydantic Basics

Pydantic models define the exact structure you want:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class ReviewAnalysis(BaseModel):
    """Analysis of a product review."""
    sentiment: str = Field(description="positive, negative, or mixed")
    rating: int = Field(ge=1, le=5, description="Rating from 1-5")
    pros: List[str] = Field(description="Positive points mentioned")
    cons: List[str] = Field(description="Negative points mentioned")
```

### Key Components

| Component | Purpose | Example |
|-----------|---------|---------|
| `BaseModel` | Base class for schemas | `class MySchema(BaseModel)` |
| `Field()` | Add description/constraints | `Field(description="...")` |
| Type hints | Define data types | `str`, `int`, `List[str]` |
| `Optional` | Fields that may be missing | `Optional[str] = None` |

### Field Constraints

```python
from pydantic import Field

class Product(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0, description="Price must be positive")
    rating: int = Field(ge=1, le=5, description="Rating 1-5")
    quantity: int = Field(default=1, ge=0)
```

| Constraint | Meaning |
|------------|---------|
| `gt` | Greater than |
| `ge` | Greater than or equal |
| `lt` | Less than |
| `le` | Less than or equal |
| `min_length` | Minimum string length |
| `max_length` | Maximum string length |

---

## 3. Using Structured Output

### The `with_structured_output()` Method

```python
from langchain import init_chat_model
from pydantic import BaseModel

class Sentiment(BaseModel):
    score: float
    label: str

model = init_chat_model("openai:gpt-4o-mini")

# Attach schema to model
structured_model = model.with_structured_output(Sentiment)

# Returns typed Sentiment object, not raw text
result = structured_model.invoke("Analyze: I love this product!")
print(result.score)  # 0.9
print(result.label)  # "positive"
```

### Benefits

| Without Structured Output | With Structured Output |
|---------------------------|------------------------|
| Raw text string | Typed Python object |
| Manual parsing needed | Automatic parsing |
| Inconsistent format | Guaranteed schema |
| No validation | Built-in validation |

---

## 4. The `response_format` Parameter

Two approaches to structured responses:

### JSON Mode (Flexible)

```python
model = init_chat_model(
    "openai:gpt-4o-mini",
    response_format={"type": "json_object"}
)

# Returns valid JSON, but shape not guaranteed
result = model.invoke("Return user info as JSON")
```

**Important:** Must mention "JSON" in your prompt!

### Structured Schema (Strict)

```python
model = init_chat_model("openai:gpt-4o-mini")
result = model.with_structured_output(MySchema).invoke(...)

# Returns exact schema every time
```

### When to Use Each

| Situation | Use |
|-----------|-----|
| Need specific fields | Structured schema |
| Flexible/dynamic structure | JSON mode |
| Programmatic processing | Structured schema |
| Quick experimentation | JSON mode |

---

## 5. Complex Nested Schemas

Pydantic supports nested structures:

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Person(BaseModel):
    name: str
    email: str
    addresses: List[Address]  # Nested!

# LLM returns properly nested structure
result = model.with_structured_output(Person).invoke(...)
print(result.addresses[0].city)  # Fully typed access
```

### Real-World Example: Meeting Extraction

```python
class Attendee(BaseModel):
    name: str
    email: Optional[str] = None
    role: Optional[str] = None

class MeetingRequest(BaseModel):
    title: str
    date: Optional[str] = None
    time: Optional[str] = None
    attendees: List[Attendee]
    priority: str = Field(description="low, medium, or high")
    agenda: List[str]
```

---

## 6. Using Enums for Constrained Values

```python
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

class ReviewAnalysis(BaseModel):
    sentiment: Sentiment  # Can only be one of the enum values
    confidence: float
```

**Benefits:**
- IDE autocomplete
- Type safety
- Clear valid options for LLM

---

## 7. Validation and Error Handling

### Automatic Validation

Pydantic validates automatically:

```python
from pydantic import ValidationError

class Rating(BaseModel):
    score: int = Field(ge=1, le=5)

try:
    rating = Rating(score=10)  # Invalid!
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Handling LLM Output Errors

```python
from pydantic import ValidationError

def safe_extract(text: str, schema):
    """Extract with error handling."""
    model = init_chat_model("openai:gpt-4o-mini")
    
    try:
        result = model.with_structured_output(schema).invoke(text)
        return {"success": True, "data": result}
    except ValidationError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

---

## 8. Fallback Strategies

When structured output fails, have a backup plan:

### Strategy 1: Simpler Schema

```python
class DetailedAnalysis(BaseModel):
    sentiment: str
    score: float
    pros: List[str]
    cons: List[str]
    summary: str
    keywords: List[str]

class SimpleAnalysis(BaseModel):
    sentiment: str
    summary: str

def analyze_with_fallback(text: str):
    # Try detailed first
    try:
        return model.with_structured_output(DetailedAnalysis).invoke(text)
    except:
        pass
    
    # Fall back to simple
    try:
        return model.with_structured_output(SimpleAnalysis).invoke(text)
    except:
        return None
```

### Strategy 2: Retry Logic

```python
def extract_with_retry(text: str, schema, max_retries: int = 3):
    """Retry extraction on failure."""
    for attempt in range(max_retries):
        try:
            return model.with_structured_output(schema).invoke(text)
        except ValidationError:
            if attempt < max_retries - 1:
                continue
            raise
```

### Strategy 3: Default Values

```python
class Analysis(BaseModel):
    sentiment: str = "unknown"
    confidence: float = 0.0
    notes: Optional[str] = None

# Fields have defaults if LLM can't determine
```

---

## 9. Best Practices

### Schema Design

| Practice | Why |
|----------|-----|
| Use clear `Field(description=...)` | Helps LLM understand what to output |
| Use `Optional` for uncertain fields | Prevents failures on missing data |
| Use Enums for fixed choices | Constrains output to valid values |
| Keep schemas focused | Don't ask for too much at once |

### Field Descriptions

```python
# ❌ Vague
class Order(BaseModel):
    amount: float
    date: str

# ✅ Clear
class Order(BaseModel):
    total_usd: float = Field(description="Total amount in US dollars")
    order_date: str = Field(description="Date in YYYY-MM-DD format")
    status: str = Field(description="One of: pending, shipped, delivered")
```

---

## 10. Common Patterns

### Pattern 1: Data Extraction

```python
class ExtractedEntities(BaseModel):
    people: List[str] = Field(description="Names of people mentioned")
    organizations: List[str] = Field(description="Company/org names")
    dates: List[str] = Field(description="Dates in YYYY-MM-DD format")
    locations: List[str] = Field(description="Places mentioned")
```

### Pattern 2: Classification

```python
class Classification(BaseModel):
    category: str = Field(description="One of: bug, feature, question, other")
    priority: str = Field(description="One of: low, medium, high, critical")
    confidence: float = Field(ge=0, le=1, description="Confidence 0-1")
```

### Pattern 3: Structured Response

```python
class TaskResult(BaseModel):
    completed: bool
    summary: str
    next_steps: List[str]
    warnings: Optional[List[str]] = None
```

---

## 11. Structured Output with Agents

Agents can use tools and return structured data:

```python
from langchain_core.tools import tool

@tool
def lookup_product(product_id: str) -> str:
    """Look up product info."""
    # Return JSON string for agent to process
    return '{"name": "Widget", "price": 29.99}'

# Agent processes tool output, can structure final response
```

---

## Key Takeaways

1. **Pydantic models define schemas** - Use `BaseModel` and `Field()`
2. **`with_structured_output()`** - Guarantees typed responses
3. **Field descriptions matter** - Help the LLM understand requirements
4. **Handle validation errors** - Always wrap in try/except
5. **Use fallbacks** - Simpler schema or retry when needed
6. **Enums constrain values** - Use for fixed choices
7. **Nested schemas work** - Build complex structures
8. **Optional fields are safer** - For data that might be missing

---

## Additional Resources

- [Pydantic Documentation](https://docs.pydantic.dev/latest/)
- [LangChain Structured Output](https://docs.langchain.com/oss/python/langchain/how-to/structured_output)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

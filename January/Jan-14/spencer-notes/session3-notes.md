# Session 3: Production Patterns

## Overview

This session covers building robust, production-ready agent systems. You'll learn error handling patterns, retry strategies, and how to connect agents to knowledge bases using RAG (Retrieval Augmented Generation).

---

## Part A: Error Handling Patterns

### 1. Why Error Handling Matters

Production agents encounter errors:
- API rate limits
- Network timeouts
- Malformed inputs
- External service failures
- Unexpected model behavior

**Without error handling:** Single failure crashes your system  
**With error handling:** Graceful degradation, continued operation

---

### 2. Common Error Categories

| Category | Examples | Strategy |
|----------|----------|----------|
| **API Errors** | Rate limits (429), auth failures (401) | Retry with backoff |
| **Tool Errors** | External API down, invalid data | Fallback, user message |
| **Validation Errors** | Output doesn't match schema | Retry or simplify |
| **Timeout Errors** | Slow responses | Retry, extend timeout |
| **Input Errors** | Malformed user input | Validate early, clarify |

---

### 3. Tool-Level Error Handling

**First line of defense:** Handle errors inside tools, return messages instead of crashing.

```python
# ❌ BAD: No error handling
@tool
def bad_api_call(endpoint: str) -> str:
    response = requests.get(endpoint)  # Can crash!
    return response.text

# ✅ GOOD: Graceful error handling
@tool
def good_api_call(endpoint: str) -> str:
    try:
        response = requests.get(endpoint, timeout=5)
        response.raise_for_status()
        return response.text
    except requests.Timeout:
        return "ERROR: Request timed out. Please try again."
    except requests.HTTPError as e:
        return f"ERROR: HTTP {e.response.status_code}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}"
```

**Key Principle:** Return error strings, don't raise exceptions.

---

### 4. Retry with Exponential Backoff

For transient failures, retry with increasing delays:

```
Attempt 1: Fail → Wait 1s
Attempt 2: Fail → Wait 2s
Attempt 3: Fail → Wait 4s
Attempt 4: Success!
```

```python
def retry_with_backoff(func, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            result = func()
            if not result.startswith("ERROR:"):
                return result
        except Exception:
            pass
        
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
    
    return "FAILED: Max retries exceeded"
```

### Backoff Guidelines

| Scenario | Base Delay | Max Retries |
|----------|------------|-------------|
| Rate limits | 1-2s | 3-5 |
| Network issues | 1s | 3 |
| External APIs | 2s | 3 |
| Database | 0.5s | 3 |

---

### 5. Fallback Mechanisms

When primary approach fails, use alternatives:

```
Primary Method
    ↓ (fails)
Fallback Method
    ↓ (fails)
Default Response
```

```python
def search_with_fallback(query: str) -> str:
    # Try primary (fast)
    result = primary_search(query)
    if not result.startswith("ERROR:"):
        return result
    
    # Try backup (slower but reliable)
    result = backup_search(query)
    if not result.startswith("ERROR:"):
        return result
    
    # Default response
    return "Search temporarily unavailable. Please try again."
```

---

### 6. Input Validation

Catch bad inputs before they cause problems:

```python
def validate_and_process(user_input: str) -> str:
    # Check length
    if len(user_input) > 10000:
        return "Message too long. Please keep under 10,000 characters."
    
    # Check empty
    if not user_input.strip():
        return "Please provide a message."
    
    # Safe to proceed
    return process(user_input)
```

---

### 7. Error Handling Best Practices

| Practice | Why |
|----------|-----|
| Validate early | Catch problems before they propagate |
| Fail gracefully | Return messages, don't crash |
| Implement retries | Transient errors often resolve |
| Provide fallbacks | Critical functions need backups |
| Log errors | Debug production issues |
| Never expose secrets | Sanitize error messages |

---

## Part B: Building RAG Tools for Agents

### 8. What is RAG?

**RAG = Retrieval Augmented Generation**

Connects agents to your knowledge base instead of relying only on training data.

```
Traditional LLM:
User Question → LLM (training data only) → Answer

RAG:
User Question → Search Knowledge Base → LLM + Retrieved Context → Answer
```

---

### 9. 2-Step RAG vs Agentic RAG

| Aspect | 2-Step RAG | Agentic RAG |
|--------|------------|-------------|
| Retrieval | ALWAYS retrieves | Agent DECIDES when |
| Query | Single, fixed | Can reformulate |
| Flow | Retrieve → Generate | Think → Maybe Retrieve → Generate |
| Complexity | Simpler | More flexible |
| Best for | Simple QA | Complex conversations |

```
2-Step RAG:
Query → [Always Search] → Generate

Agentic RAG:
Query → [Agent Decides] → Search? → Generate
                       → No Search → Generate directly
```

---

### 10. Creating a RAG Tool

Wrap your vector store in a tool:

```python
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search company knowledge base for relevant information.
    
    Use for questions about:
    - Company policies
    - Product information
    - Technical documentation
    """
    # Search vector store
    docs = vectorstore.similarity_search(query, k=3)
    
    if not docs:
        return "No relevant information found."
    
    # Format results
    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"[{i}] {doc.page_content}")
    
    return "\n\n".join(results)
```

**Key Elements:**
- Clear docstring tells agent WHEN to use it
- Returns formatted text for agent to synthesize
- Handles empty results gracefully

---

### 11. Building a RAG Agent

```python
rag_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_knowledge_base],
    system_prompt="""You are a support agent with knowledge base access.
    
    When answering questions:
    1. Search the knowledge base for relevant information
    2. Synthesize into a helpful response
    3. Cite sources when providing policy details
    4. If not found, acknowledge and offer alternatives
    """,
    checkpointer=InMemorySaver(),
    name="rag_agent"
)
```

---

### 12. RAG Tool Best Practices

| Practice | Why |
|----------|-----|
| Limit chunk size | Prevent token overflow |
| Include source metadata | Enable citations |
| Filter by relevance score | Avoid low-quality results |
| Clear tool description | Help agent choose correctly |
| Handle empty results | Graceful "not found" |
| Multiple specific tools | Better than one generic |

---

### 13. Enhanced RAG with Metadata

```python
@tool
def search_with_sources(query: str) -> str:
    """Search and return results with source citations."""
    results = vectorstore.similarity_search(query, k=3)
    
    if not results:
        return "No relevant information found."
    
    formatted = []
    for doc in results:
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    
    return "\n\n---\n\n".join(formatted)
```

---

### 14. Multi-Source RAG

Agents can use multiple retrieval tools:

```python
@tool
def search_policies(query: str) -> str:
    """Search HR and company policies."""
    ...

@tool
def search_products(query: str) -> str:
    """Search product catalog."""
    ...

@tool
def search_support(query: str) -> str:
    """Search support tickets and solutions."""
    ...

# Agent chooses the right source based on question
agent = create_agent(
    tools=[search_policies, search_products, search_support],
    ...
)
```

---

### 15. Production RAG Checklist

```
☐ USE REAL VECTOR STORE
  ChromaDB, Pinecone, etc. (not simulation)

☐ QUALITY EMBEDDINGS
  text-embedding-3-small or better
  Same model for indexing AND querying

☐ APPROPRIATE CHUNK SIZE
  800-1200 characters typical
  ~200 character overlap

☐ RETRIEVAL TUNING
  Adjust k based on use case (3-5 typical)
  Score thresholding for quality

☐ ERROR HANDLING
  Handle empty results
  Timeout for vector store queries
  Fallback responses

☐ MONITORING
  Log retrieval quality
  Track which documents are used
  Monitor latency
```

---

### 16. Combining Error Handling + RAG

Production RAG tools need robust error handling:

```python
@tool
def robust_search(query: str) -> str:
    """Search with error handling."""
    try:
        # Validate input
        if not query or len(query) < 2:
            return "Please provide a more specific query."
        
        # Search with timeout
        docs = vectorstore.similarity_search(query, k=3)
        
        if not docs:
            return "No relevant documents found for your query."
        
        # Format results
        return format_results(docs)
        
    except TimeoutError:
        return "Search timed out. Please try again."
    except Exception as e:
        return f"Search unavailable. Please try again later."
```

---

## Key Takeaways

### Error Handling
1. **Handle errors in tools** - Return messages, don't crash
2. **Retry with backoff** - Transient errors often resolve
3. **Implement fallbacks** - Critical functions need backups
4. **Validate inputs early** - Catch problems before they propagate
5. **Log everything** - Debug production issues

### RAG
1. **RAG connects agents to knowledge** - Beyond training data
2. **Tool docstrings guide usage** - Tell agent when to search
3. **Agentic RAG is flexible** - Agent decides when to retrieve
4. **Format results for agents** - Include metadata, limit size
5. **Multiple tools for multiple sources** - Better than generic search

---

## Additional Resources

- [LangChain Error Handling](https://docs.langchain.com/oss/python/langchain/how-to/error_handling)
- [LangChain RAG Tutorial](https://docs.langchain.com/oss/python/langchain/tutorials/rag)
- [Vector Stores](https://docs.langchain.com/oss/python/langchain/integrations/vectorstores)
- [Retry Libraries (tenacity)](https://github.com/jd/tenacity)

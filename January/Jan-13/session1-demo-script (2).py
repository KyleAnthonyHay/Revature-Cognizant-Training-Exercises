"""
Session 1 Demo: Foundation & Core Concepts
==========================================

This demo script covers:
1. Environment setup verification
2. init_chat_model() basics with AWS Bedrock
3. Model configuration (temperature, max_tokens)
4. Invocation patterns (invoke, batch, stream)

Prerequisites:
- Python 3.9+
- pip install langchain langchain-aws python-dotenv boto3
- AWS credentials configured

Run this script section by section during the training session.

LangChain Version: v1.0+
Last Verified: January 2026
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the same directory as this script
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# ============================================================================
# AWS BEDROCK CONFIGURATION - From .env file
# ============================================================================

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")

# Bedrock Model ID
BEDROCK_MODEL = "bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0"
# Alternative models:
# BEDROCK_MODEL = "bedrock:anthropic.claude-3-haiku-20240307-v1:0"  # Faster, cheaper
# BEDROCK_MODEL = "bedrock:amazon.titan-text-express-v1"            # Amazon Titan


# ============================================================================
# PART 1: Environment Setup Verification
# ============================================================================

def part1_environment_check():
    """Verify AWS credentials and Bedrock access."""
    
    print("=" * 70)
    print("PART 1: Environment Setup Verification (AWS Bedrock)")
    print("=" * 70)
    
    print("""
    AWS Bedrock Configuration:
    - AWS_ACCESS_KEY_ID: Your AWS access key
    - AWS_SECRET_ACCESS_KEY: Your AWS secret key
    - AWS_DEFAULT_REGION: Region where Bedrock is enabled (e.g., us-east-1)
    """)
    
    # Check AWS variables
    print("\n[Step 1] Checking AWS Configuration")
    print("-" * 40)
    
    aws_vars = {
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION")
    }
    
    all_set = True
    for var, value in aws_vars.items():
        if value and value != "your-access-key-id" and value != "your-secret-access-key":
            if "SECRET" in var:
                display = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            else:
                display = value
            print(f"  ✓ {var} = {display}")
        else:
            print(f"  ✗ {var} = NOT SET or using placeholder")
            all_set = False
    
    if not all_set:
        print("\n  ⚠️  Please update AWS credentials at the top of this script!")
        print("  Replace 'your-access-key-id' and 'your-secret-access-key' with real values.")
        return False
    
    # Test Bedrock connection
    print("\n[Step 2] Testing Bedrock Connection")
    print("-" * 40)
    
    try:
        from langchain.chat_models import init_chat_model
        
        model = init_chat_model(BEDROCK_MODEL)
        response = model.invoke("Say 'Bedrock connection successful!' in exactly those words.")
        print(f"  ✅ Bedrock connected successfully!")
        print(f"  Model: {BEDROCK_MODEL}")
        print(f"  Response: {response.content[:50]}...")
        return True
        
    except Exception as e:
        print(f"  ❌ Bedrock connection failed: {e}")
        print("\n  Troubleshooting:")
        print("  1. Verify AWS credentials are correct")
        print("  2. Ensure Bedrock is enabled in your AWS region")
        print("  3. Check that you have model access in Bedrock console")
        return False


# ============================================================================
# PART 2: init_chat_model() Basics with Bedrock
# ============================================================================

def part2_init_chat_model_basics():
    """Demonstrate basic model initialization with init_chat_model() and Bedrock."""
    
    print("\n" + "=" * 70)
    print("PART 2: init_chat_model() with AWS Bedrock")
    print("=" * 70)
    
    print("""
    Key Concept: init_chat_model() works with ANY provider including Bedrock
    
    Format: "provider:model-id"
    
    Bedrock Examples:
      - "bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0"  (Claude 3.5 Sonnet)
      - "bedrock:anthropic.claude-3-haiku-20240307-v1:0"    (Claude 3 Haiku - fast)
      - "bedrock:amazon.titan-text-express-v1"              (Amazon Titan)
      - "bedrock:meta.llama3-1-70b-instruct-v1:0"          (Llama 3.1)
    """)
    
    from langchain.chat_models import init_chat_model
    
    # Step 1: Basic initialization
    print("\n[Step 3] Basic Model Initialization")
    print("-" * 40)
    
    model = init_chat_model(BEDROCK_MODEL)
    print(f"  Model created: {type(model).__name__}")
    print(f"  Provider string: '{BEDROCK_MODEL}'")
    
    # Step 2: Simple invocation
    print("\n[Step 4] Simple Invocation")
    print("-" * 40)
    
    response = model.invoke("What is 2 + 2? Reply with just the number.")
    print(f"  Question: What is 2 + 2?")
    print(f"  Response: {response.content}")
    
    # Step 3: Message format
    print("\n[Step 5] Using Message Format (Recommended)")
    print("-" * 40)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    response = model.invoke(messages)
    print(f"  System: 'You are a helpful assistant. Be concise.'")
    print(f"  User: 'What is the capital of France?'")
    print(f"  Assistant: {response.content}")
    
    print("\n  ✅ init_chat_model() provides a unified interface - same code works")
    print("     with OpenAI, Anthropic, Bedrock, and other providers!")
    
    return model


# ============================================================================
# PART 3: Model Configuration
# ============================================================================

def part3_model_configuration():
    """Demonstrate temperature and max_tokens configuration."""
    
    print("\n" + "=" * 70)
    print("PART 3: Model Configuration Options")
    print("=" * 70)
    
    print("""
    Key Parameters:
      - temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
      - max_tokens: Limits response length
    """)
    
    from langchain.chat_models import init_chat_model
    
    # Temperature demonstration
    print("\n[Step 6] Temperature Effect on Responses")
    print("-" * 40)
    
    # Deterministic model
    model_deterministic = init_chat_model(
        BEDROCK_MODEL,
        temperature=0.0
    )
    
    # Creative model
    model_creative = init_chat_model(
        BEDROCK_MODEL,
        temperature=1.0
    )
    
    prompt = "Name a color. Reply with just one word."
    
    print(f"\n  Prompt: '{prompt}'")
    print("\n  Temperature = 0.0 (deterministic - same answer each time):")
    for i in range(3):
        resp = model_deterministic.invoke(prompt)
        print(f"    Run {i+1}: {resp.content.strip()}")
    
    print("\n  Temperature = 1.0 (creative - varied answers):")
    for i in range(3):
        resp = model_creative.invoke(prompt)
        print(f"    Run {i+1}: {resp.content.strip()}")
    
    # Max tokens demonstration
    print("\n[Step 7] Limiting Response Length with max_tokens")
    print("-" * 40)
    
    model_short = init_chat_model(
        BEDROCK_MODEL,
        max_tokens=30
    )
    
    model_long = init_chat_model(
        BEDROCK_MODEL,
        max_tokens=150
    )
    
    explain_prompt = "Explain what machine learning is."
    
    print(f"\n  Prompt: '{explain_prompt}'")
    
    print("\n  max_tokens=30 (short response):")
    short_response = model_short.invoke(explain_prompt)
    print(f"    {short_response.content}")
    
    print("\n  max_tokens=150 (longer response):")
    long_response = model_long.invoke(explain_prompt)
    print(f"    {long_response.content}")
    
    print("\n  💡 Tip: Use max_tokens to control costs and ensure concise responses!")


# ============================================================================
# PART 4: Invocation Patterns - invoke() and batch()
# ============================================================================

def part4_invoke_and_batch():
    """Compare invoke() vs batch() performance."""
    
    print("\n" + "=" * 70)
    print("PART 4: Invocation Patterns - invoke() vs batch()")
    print("=" * 70)
    
    print("""
    Three main patterns:
      .invoke()  - Single synchronous request (blocks until complete)
      .batch()   - Multiple requests in parallel (MUCH faster!)
      .stream()  - Real-time token output (great for chat UX)
    
    Let's compare invoke() in a loop vs batch()...
    """)
    
    from langchain.chat_models import init_chat_model
    
    model = init_chat_model(BEDROCK_MODEL, temperature=0)
    
    # Prepare questions
    questions = [
        "What is the capital of France? One word answer.",
        "What is the capital of Germany? One word answer.",
        "What is the capital of Italy? One word answer.",
        "What is the capital of Spain? One word answer.",
        "What is the capital of Japan? One word answer.",
    ]
    
    # Sequential approach (SLOW)
    print("\n[Step 8] Sequential .invoke() calls (the WRONG way)")
    print("-" * 40)
    
    start_time = time.time()
    sequential_results = []
    for q in questions:
        response = model.invoke(q)
        sequential_results.append(response.content.strip())
    sequential_time = time.time() - start_time
    
    print(f"  Questions processed: {len(questions)}")
    print(f"  Time taken: {sequential_time:.2f} seconds")
    
    # Batch approach (FAST)
    print("\n[Step 9] Single .batch() call (the RIGHT way)")
    print("-" * 40)
    
    start_time = time.time()
    batch_results = model.batch(questions)
    batch_time = time.time() - start_time
    
    print(f"  Questions processed: {len(questions)}")
    print(f"  Time taken: {batch_time:.2f} seconds")
    
    # Comparison
    speedup = sequential_time / batch_time if batch_time > 0 else 0
    print(f"\n  🚀 Speedup: {speedup:.1f}x faster with batch()!")
    
    # Show results
    print("\n[Results]")
    print("-" * 40)
    for q, r in zip(questions, batch_results):
        country = q.split("capital of ")[1].split("?")[0]
        print(f"  {country}: {r.content.strip()}")
    
    print("""
    
    ⚠️  IMPORTANT: Never use invoke() in a loop when you can use batch()!
    
    Rule of thumb:
      - 1 request → use .invoke()
      - 2+ independent requests → use .batch()
    """)


# ============================================================================
# PART 5: Streaming Pattern
# ============================================================================

def part5_streaming():
    """Demonstrate real-time streaming output."""
    
    print("\n" + "=" * 70)
    print("PART 5: Streaming - Real-Time Output")
    print("=" * 70)
    
    print("""
    .stream() yields tokens as they're generated:
      - Shows output in real-time (like ChatGPT typing)
      - Great for chat interfaces and long responses
      - Same total time, but better perceived performance
    """)
    
    from langchain.chat_models import init_chat_model
    
    model = init_chat_model(BEDROCK_MODEL)
    
    # Streaming demonstration
    print("\n[Step 10] Watch the streaming effect")
    print("-" * 40)
    print("\n  Generating a short story (watch it appear!):\n")
    print("  ", end="", flush=True)
    
    prompt = "Write a very short story about a robot learning to paint. Keep it under 75 words."
    
    for chunk in model.stream(prompt):
        print(chunk.content, end="", flush=True)
    
    print("\n")
    
    # Time to first token comparison
    print("\n[Step 11] Time-to-First-Token Comparison")
    print("-" * 40)
    
    test_prompt = "Count from 1 to 10, with each number on a new line."
    
    # With invoke
    print("\n  Using .invoke() (wait for complete response):")
    start = time.time()
    response = model.invoke(test_prompt)
    total_time = time.time() - start
    print(f"    Time to see anything: {total_time:.2f}s")
    
    # With stream
    print("\n  Using .stream() (see tokens immediately):")
    start = time.time()
    first_token_time = None
    full_response = ""
    
    for chunk in model.stream(test_prompt):
        if first_token_time is None:
            first_token_time = time.time() - start
        full_response += chunk.content
    
    total_stream_time = time.time() - start
    print(f"    Time to first token: {first_token_time:.2f}s")
    print(f"    Time to complete: {total_stream_time:.2f}s")
    
    print("""
    
    💡 Insight: Total time is similar, but streaming feels faster!
       Users prefer seeing progress over waiting for complete output.
    """)


# ============================================================================
# PART 6: Summary
# ============================================================================

def part6_summary():
    """Summary and pattern selection guide."""
    
    print("\n" + "=" * 70)
    print("PART 6: Summary - Choosing the Right Pattern")
    print("=" * 70)
    
    print(f"""
    ┌─────────────────┬─────────────────────────────────────────────────────┐
    │ Pattern         │ When to Use                                         │
    ├─────────────────┼─────────────────────────────────────────────────────┤
    │ .invoke()       │ Single request, need complete response before       │
    │                 │ continuing, simple Q&A interactions                 │
    ├─────────────────┼─────────────────────────────────────────────────────┤
    │ .batch()        │ Multiple independent requests, bulk processing,    │
    │                 │ ALWAYS prefer over invoke() in a loop!              │
    ├─────────────────┼─────────────────────────────────────────────────────┤
    │ .stream()       │ Chat interfaces, long responses, when users want   │
    │                 │ to see progress (typing effect)                     │
    └─────────────────┴─────────────────────────────────────────────────────┘
    
    Decision Tree:
    
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
    
    
    Key Takeaways from Session 1:
    ═════════════════════════════
    
    1. ✅ Use init_chat_model("provider:model-name") for all models
    2. ✅ Bedrock format: "bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0"
    3. ✅ temperature=0 for deterministic, higher for creative
    4. ✅ batch() is faster than invoke() loops
    5. ✅ stream() improves perceived performance
    6. ❌ Avoid deprecated patterns (LCEL, create_react_agent)
    
    Current Configuration:
    ══════════════════════
    Model: {BEDROCK_MODEL}
    Region: {AWS_REGION}
    
    Coming in Session 2: Building Agents with Tools!
    """)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_full_demo():
    """Run all parts of the demo sequentially."""
    
    print("\n" + "=" * 70)
    print("   SESSION 1 DEMO: Foundation & Core Concepts")
    print("   LangChain v1.0 Training with AWS Bedrock")
    print("=" * 70)
    
    # Part 1: Environment check
    if not part1_environment_check():
        print("\n⚠️  Fix AWS configuration before continuing.")
        print("   Update the credentials at the top of this script.")
        return
    
    input("\nPress Enter to continue to Part 2 (init_chat_model basics)...")
    part2_init_chat_model_basics()
    
    input("\nPress Enter to continue to Part 3 (model configuration)...")
    part3_model_configuration()
    
    input("\nPress Enter to continue to Part 4 (invoke vs batch)...")
    part4_invoke_and_batch()
    
    input("\nPress Enter to continue to Part 5 (streaming)...")
    part5_streaming()
    
    input("\nPress Enter to see the summary...")
    part6_summary()
    
    print("\n" + "=" * 70)
    print("   DEMO COMPLETE!")
    print("=" * 70)
    print("\n   Questions? Let's discuss!\n")


def run_part(part_number: int):
    """Run a specific part of the demo."""

    parts = {
        1: part1_environment_check,
        2: part2_init_chat_model_basics,
        3: part3_model_configuration,
        4: part4_invoke_and_batch,
        5: part5_streaming,
        6: part6_summary
    }
    
    if part_number in parts:
        parts[part_number]()
    else:
        print(f"Invalid part number. Choose 1-6.")


if __name__ == "__main__":
    run_full_demo()

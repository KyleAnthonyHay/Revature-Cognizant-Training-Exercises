"""
Session 2: Structured Output & Validation - Demo Code

This demo covers:
- Part 1: Unstructured output (the problem)
- Part 2: Pydantic schemas basics
- Part 3: Using with_structured_output()
- Part 4: Complex nested schemas
- Part 5: Enums for constrained values
- Part 6: Validation and error handling
- Part 7: Fallback strategies

Prerequisites:
    pip install langchain langchain-openai pydantic langsmith

Usage:
    Comment/uncomment the function calls in main() to run specific demos.
"""

from langchain.agents import create_agent
import os
from dotenv import load_dotenv

# =============================================================================
# API KEYS AND CONFIGURATION
# =============================================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "session1-agent-memory")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")

if not LANGSMITH_API_KEY:
    raise ValueError("LANGSMITH_API_KEY not found in environment variables. Please set it in .env file.")

# Set environment variables for LangChain/LangSmith
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
os.environ["LANGSMITH_TRACING"] = "true"

# =============================================================================

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from enum import Enum


# ============================================================================
# PART 1: Unstructured Output (The Problem)
# ============================================================================

def part1_unstructured_output():
    """Demonstrates the problem with raw text output."""
    
    print("=" * 60)
    print("PART 1: Unstructured Output (The Problem)")
    print("=" * 60)

    model = ChatOpenAI(model="gpt-4o-mini")
    
    review = "Great laptop, fast shipping, but battery life is poor. Screen is amazing!"
    
    # Unstructured request - just raw text
    response = model.invoke(f"Analyze this product review: '{review}'")
    
    print(f"\nReview: {review}")
    print(f"\nRaw Response (unstructured):")
    print(f"  {response.content[:200]}...")
    print(f"\n  Type: {type(response.content)}")
    print("\n[!] Problem: How do we extract sentiment, rating, pros/cons programmatically?")


# ============================================================================
# PART 2: Pydantic Schema Basics
# ============================================================================

def part2_pydantic_basics():
    """Demonstrates basic Pydantic schema definition."""
    
    print("=" * 60)
    print("PART 2: Pydantic Schema Basics")
    print("=" * 60)

    # Define a simple schema
    class ProductReview(BaseModel):
        """Schema for product review analysis."""
        sentiment: str = Field(description="Overall sentiment: positive, negative, or mixed")
        rating: int = Field(ge=1, le=5, description="Star rating from 1-5")
        summary: str = Field(description="One sentence summary")

    print("\nSchema: ProductReview")
    print(f"  Fields: {list(ProductReview.model_fields.keys())}")
    
    # Manual instantiation works
    review = ProductReview(
        sentiment="positive",
        rating=4,
        summary="Good product with minor issues"
    )
    print(f"\nManual instance:")
    print(f"  sentiment: {review.sentiment}")
    print(f"  rating: {review.rating}")
    print(f"  summary: {review.summary}")
    
    # Validation catches errors
    print("\nValidation demo:")
    try:
        bad_review = ProductReview(sentiment="good", rating=10, summary="Test")
        print(f"  Created: {bad_review}")
    except ValidationError as e:
        print(f"  [!] ValidationError: rating=10 is > 5 (max)")


# ============================================================================
# PART 3: Using with_structured_output()
# ============================================================================

def part3_structured_output():
    """Demonstrates with_structured_output() for typed responses."""
    
    print("=" * 60)
    print("PART 3: Using with_structured_output()")
    print("=" * 60)

    # Define schema
    class ReviewAnalysis(BaseModel):
        """Structured analysis of a product review."""
        sentiment: str = Field(description="positive, negative, or mixed")
        rating: int = Field(ge=1, le=5, description="Estimated star rating 1-5")
        pros: List[str] = Field(description="List of positive points")
        cons: List[str] = Field(description="List of negative points")
        summary: str = Field(description="One sentence summary")

    model = ChatOpenAI(model="gpt-4o-mini")
    
    # Attach schema to model
    structured_model = model.with_structured_output(ReviewAnalysis)
    
    review = "Great laptop, fast shipping, but battery life is poor. Screen is amazing!"
    
    # Get structured response
    result = structured_model.invoke(f"Analyze this review: '{review}'")
    
    print(f"\nReview: {review}")
    print(f"\nStructured Response:")
    print(f"  Type: {type(result).__name__}")
    print(f"  sentiment: {result.sentiment}")
    print(f"  rating: {result.rating}/5")
    print(f"  pros: {result.pros}")
    print(f"  cons: {result.cons}")
    print(f"  summary: {result.summary}")
    print("\n[✓] Clean, typed, validated output!")


# ============================================================================
# PART 4: Complex Nested Schemas
# ============================================================================

def part4_nested_schemas():
    """Demonstrates nested Pydantic schemas."""
    
    print("=" * 60)
    print("PART 4: Complex Nested Schemas")
    print("=" * 60)

    # Nested schema definitions
    class Attendee(BaseModel):
        """Person attending a meeting."""
        name: str = Field(description="Person's full name")
        email: Optional[str] = Field(default=None, description="Email if mentioned")
        department: Optional[str] = Field(default=None, description="Department if mentioned")

    class MeetingRequest(BaseModel):
        """Extracted meeting request details."""
        title: str = Field(description="Meeting topic or title")
        date: Optional[str] = Field(default=None, description="Proposed date (YYYY-MM-DD)")
        time: Optional[str] = Field(default=None, description="Proposed time (HH:MM)")
        attendees: List[Attendee] = Field(description="People to invite")
        priority: str = Field(description="low, medium, or high")
        action_items: List[str] = Field(description="Tasks or agenda items")

    model = ChatOpenAI(model="gpt-4o-mini")
    structured_model = model.with_structured_output(MeetingRequest)
    
    
    email_text = """
    Hey team,
    
    Can we set up a call to discuss the Q4 roadmap? I'm thinking next Tuesday at 2pm.
    Please include John Smith (john@company.com) from Engineering and 
    Sarah Johnson from Marketing.
    
    This is urgent - we need to finalize before the board meeting.
    
    Agenda:
    - Review Q3 results
    - Set Q4 targets
    - Assign owners
    """
    
    result = model.invoke(f"Extract meeting details:\n\n{email_text}")
    print(result)
    
    print(f"\nExtracted Meeting Request:")
    print(f"  title: {result.title}")
    print(f"  date: {result.date}")
    print(f"  time: {result.time}")
    print(f"  priority: {result.priority}")
    print(f"\n  Attendees ({len(result.attendees)}):")
    for att in result.attendees:
        print(f"    - {att.name} | {att.email} | {att.department}")
    print(f"\n  Action Items:")
    for item in result.action_items:
        print(f"    - {item}")


# ============================================================================
# PART 5: Enums for Constrained Values
# ============================================================================

def part5_enum_constraints():
    """Demonstrates using Enums to constrain output values."""
    
    print("=" * 60)
    print("PART 5: Enums for Constrained Values")
    print("=" * 60)

    # Define enums for constrained choices
    class Sentiment(str, Enum):
        POSITIVE = "positive"
        NEGATIVE = "negative"
        NEUTRAL = "neutral"
        MIXED = "mixed"

    class Priority(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"

    class TicketClassification(BaseModel):
        """Support ticket classification."""
        sentiment: Sentiment = Field(description="Customer sentiment")
        priority: Priority = Field(description="Ticket priority")
        category: str = Field(description="bug, feature, question, or other")
        summary: str = Field(description="Brief summary of the issue")

    model = ChatOpenAI(model="gpt-4o-mini")
    structured_model = model.with_structured_output(TicketClassification)
    
    tickets = [
        "My app keeps crashing! I've lost all my work. This is unacceptable!",
        "Would be nice to have dark mode. Just a suggestion for future updates.",
        "How do I export my data to CSV? Can't find the option."
    ]
    
    print("\nClassifying support tickets:")
    print("-" * 40)
    
    for ticket in tickets:
        result = structured_model.invoke(f"Classify this ticket: '{ticket}'")
        print(f"\nTicket: {ticket[:50]}...")
        print(f"  sentiment: {result.sentiment.value}")
        print(f"  priority: {result.priority.value}")
        print(f"  category: {result.category}")


# ============================================================================
# PART 6: Validation and Error Handling
# ============================================================================

def part6_validation_error_handling():
    """Demonstrates validation and error handling patterns."""
    
    print("=" * 60)
    print("PART 6: Validation and Error Handling")
    print("=" * 60)

    class StrictRating(BaseModel):
        """Rating with strict constraints."""
        score: int = Field(ge=1, le=5, description="Rating 1-5 only")
        reason: str = Field(min_length=10, description="Reason (min 10 chars)")

    print("\nValidation Demo:")
    print("-" * 40)
    
    # Valid case
    try:
        valid = StrictRating(score=4, reason="Great product with fast delivery!")
        print(f"[✓] Valid: score={valid.score}, reason='{valid.reason[:20]}...'")
    except ValidationError as e:
        print(f"[✗] Error: {e}")
    
    # Invalid: score out of range
    try:
        invalid = StrictRating(score=10, reason="Testing")
        print(f"[✓] Valid: score={invalid.score}")
    except ValidationError as e:
        print(f"[✗] Invalid score: score=10 exceeds maximum of 5")
    
    # Invalid: reason too short
    try:
        invalid = StrictRating(score=3, reason="Short")
        print(f"[✓] Valid: reason='{invalid.reason}'")
    except ValidationError as e:
        print(f"[✗] Invalid reason: 'Short' is less than 10 characters")

    # Safe extraction function
    def safe_extract(text: str, schema, model):
        """Extract with error handling."""
        try:
            result = model.with_structured_output(schema).invoke(text)
            return {"success": True, "data": result}
        except ValidationError as e:
            return {"success": False, "error": f"Validation: {e.error_count()} errors"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    print("\nSafe Extraction Demo:")
    print("-" * 40)
    
    model = ChatOpenAI(model="gpt-4o-mini")
    
    result = safe_extract("Rate this product: Amazing!", StrictRating, model)
    if result["success"]:
        print(f"[✓] Extracted: score={result['data'].score}")
    else:
        print(f"[✗] Failed: {result['error']}")


# ============================================================================
# PART 7: Fallback Strategies
# ============================================================================

def part7_fallback_strategies():
    """Demonstrates fallback strategies for robust extraction."""
    
    print("=" * 60)
    print("PART 7: Fallback Strategies")
    print("=" * 60)

    # Detailed schema (might fail on vague input)
    class DetailedAnalysis(BaseModel):
        sentiment: str
        score: float = Field(ge=0, le=1)
        pros: List[str]
        cons: List[str]
        keywords: List[str]
        recommendation: str

    # Simple schema (fallback)
    class SimpleAnalysis(BaseModel):
        sentiment: str = Field(description="positive, negative, neutral, or mixed")
        summary: str = Field(description="Brief summary")

    model = ChatOpenAI(model="gpt-4o-mini")

    def analyze_with_fallback(text: str) -> dict:
        """Try detailed extraction, fall back to simple."""
        
        # Try detailed first
        try:
            result = model.with_structured_output(DetailedAnalysis).invoke(
                f"Analyze in detail: '{text}'"
            )
            return {"level": "detailed", "data": result}
        except Exception as e:
            print(f"  [!] Detailed failed: {type(e).__name__}")
        
        # Fall back to simple
        try:
            result = model.with_structured_output(SimpleAnalysis).invoke(
                f"Briefly analyze: '{text}'"
            )
            return {"level": "simple", "data": result}
        except Exception as e:
            print(f"  [!] Simple failed: {type(e).__name__}")
        
        # Last resort
        return {"level": "failed", "data": None}

    print("\nFallback Demo:")
    print("-" * 40)
    
    # Clear input - detailed should work
    clear_text = "Excellent product! Fast shipping, great quality, reasonable price. Highly recommend!"
    result = analyze_with_fallback(clear_text)
    print(f"\nClear input: '{clear_text[:40]}...'")
    print(f"  Level: {result['level']}")
    if result['data']:
        print(f"  Sentiment: {result['data'].sentiment}")
    
    # Vague input - might need fallback
    vague_text = "meh"
    result = analyze_with_fallback(vague_text)
    print(f"\nVague input: '{vague_text}'")
    print(f"  Level: {result['level']}")
    if result['data']:
        print(f"  Sentiment: {result['data'].sentiment}")


# ============================================================================
# BONUS: Structured Output with Multiple Examples
# ============================================================================

def bonus_batch_extraction():
    """Demonstrates processing multiple items with structured output."""
    
    print("=" * 60)
    print("BONUS: Batch Extraction")
    print("=" * 60)

    class ProductInfo(BaseModel):
        """Extracted product information."""
        name: str
        price: Optional[float] = None
        category: str
        in_stock: bool

    model = ChatOpenAI(model="gpt-4o-mini")
    structured_model = model.with_structured_output(ProductInfo)

    product_descriptions = [
        "The UltraBook Pro laptop ($999) is our flagship computer, currently available.",
        "Wireless earbuds - Premium audio quality, $79, temporarily out of stock.",
        "Ergonomic keyboard for programmers, priced at $149, ships immediately."
    ]

    print("\nExtracting product info from descriptions:")
    print("-" * 40)
    
    for desc in product_descriptions:
        result = structured_model.invoke(f"Extract product info: '{desc}'")
        stock_status = "✓ In Stock" if result.in_stock else "✗ Out of Stock"
        price_str = f"${result.price}" if result.price else "N/A"
        print(f"\n  {result.name}")
        print(f"    Category: {result.category}")
        print(f"    Price: {price_str}")
        print(f"    Status: {stock_status}")


# ============================================================================
# MAIN - Comment/Uncomment to run specific demos
# ============================================================================

def main():
    """
    Main entry point.
    Comment/uncomment the function calls below to run specific demos.
    """
    
    # Part 1: Unstructured output (shows the problem)
    # part1_unstructured_output()
    
    # Part 2: Pydantic schema basics
    # part2_pydantic_basics()
    
    # Part 3: Using with_structured_output()
    part3_structured_output()
    
    # Part 4: Complex nested schemas
    # part4_nested_schemas()
    
    # Part 5: Enums for constrained values
    # part5_enum_constraints()
    
    # Part 6: Validation and error handling
    # part6_validation_error_handling()
    
    # Part 7: Fallback strategies
    # part7_fallback_strategies()
    
    # Bonus: Batch extraction
    # bonus_batch_extraction()
    
    print("\n[!] Uncomment the demos you want to run in main()")


if __name__ == "__main__":
    main()

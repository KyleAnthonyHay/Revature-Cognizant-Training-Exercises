"""
Structured Output with Pydantic Demo
LangChain Version: v1.0+
"""
import os
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  ERROR: OPENAI_API_KEY environment variable is not set.")
    print("\nTo fix this, either:")
    print("  1. Set it in your terminal: export OPENAI_API_KEY='your-key-here'")
    print("  2. Create a .env file with: OPENAI_API_KEY=your-key-here")
    print("  3. Or pass it directly to init_chat_model()")
    exit(1)

# Define structured output schemas

class MovieReview(BaseModel):
    """A structured movie review analysis."""
    title: str = Field(description="The movie title")
    sentiment: Literal["positive", "negative", "mixed"] = Field(
        description="Overall sentiment of the review"
    )
    rating: float = Field(description="Rating from 0-10")
    pros: List[str] = Field(description="Positive aspects mentioned")
    cons: List[str] = Field(description="Negative aspects mentioned")
    recommended: bool = Field(description="Whether the reviewer recommends the movie")

class TaskExtraction(BaseModel):
    """Extracted tasks from a text."""
    tasks: List[str] = Field(description="List of specific tasks to complete")
    priority_task: Optional[str] = Field(
        None, description="The most important task, if identifiable"
    )
    estimated_duration: Optional[str] = Field(
        None, description="Estimated total time to complete all tasks"
    )

# Create model
try:
    model = init_chat_model("openai:gpt-4o-mini", temperature=0)
except Exception as e:
    print(f"❌ Error initializing model: {e}")
    print("\nMake sure OPENAI_API_KEY is set correctly.")
    exit(1)

# Example 1: Movie review analysis
print("=== Movie Review Analysis ===\n")

review_text = """
I watched "The Matrix" last night and it blew my mind! The action sequences 
are incredible and Keanu Reeves is perfect as Neo. The philosophical themes 
about reality are thought-provoking. However, the pacing in the middle was 
a bit slow and some of the dialogue felt wooden. Overall, a must-watch 
classic that holds up after all these years.
"""

review_analysis = model.with_structured_output(MovieReview).invoke([
    {"role": "user", "content": f"Analyze this review:\n\n{review_text}"}
])

print(f"Title: {review_analysis.title}")
print(f"Sentiment: {review_analysis.sentiment}")
print(f"Rating: {review_analysis.rating}/10")
print(f"Pros: {', '.join(review_analysis.pros)}")
print(f"Cons: {', '.join(review_analysis.cons)}")
print(f"Recommended: {'Yes' if review_analysis.recommended else 'No'}")
print()

# Example 2: Task extraction
print("=== Task Extraction ===\n")

email_text = """
Hi,

After our meeting, here's what we need to do:
- Update the proposal with Q4 projections by Friday
- Schedule a follow-up call with the client
- Review the competitor analysis Sarah shared
- Send the contract to legal for review

The proposal update is urgent since the client is waiting.

Thanks!
"""

extracted = model.with_structured_output(TaskExtraction).invoke([
    {"role": "user", "content": f"Extract tasks from this email:\n\n{email_text}"}
])

print(f"Tasks found: {len(extracted.tasks)}")
for i, task in enumerate(extracted.tasks, 1):
    print(f"  {i}. {task}")
print(f"\nPriority task: {extracted.priority_task}")
print(f"Estimated duration: {extracted.estimated_duration}")
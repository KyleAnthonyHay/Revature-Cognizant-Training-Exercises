"""
Women's Review Search Agent - ChromaDB Cloud Demo
"""
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from langchain.agents import create_agent
from tools import search_by_category, search_by_topic, search_combined, search_by_rating

BEDROCK_MODEL = "bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0"

SYSTEM_PROMPT = """You are a helpful assistant that searches women's clothing reviews.

Available categories: Intimates, Dresses, Pants, Knits, Skirts

Available tools:
- search_by_category: Get all reviews for a category
- search_by_topic: Semantic search by topic/meaning
- search_combined: Semantic search filtered by category
- search_by_rating: Filter reviews by minimum rating (1-5)

Be concise in your responses. Always use the appropriate tool based on what the user asks."""


def main():
    print("Initializing Review Search Agent...")
    #  Create model using langchain that specifies temp and max tokens
    model = init_chat_model(
        BEDROCK_MODEL,
        temperature=1.0,
        max_tokens=1000
    )


    agent = create_agent(
        model=model,
        tools=[search_by_category, search_by_topic, search_combined, search_by_rating],
        system_prompt=SYSTEM_PROMPT,
        name="review_agent",
    )
    
    print("\nAgent ready! Ask me about women's clothing reviews.")
    print("Categories: Intimates, Dresses, Pants, Knits, Skirts")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_input}]
        })
        
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"[Tool: {tc['name']}] args: {tc['args']}")
        
        response = result["messages"][-1].content
        print(f"Agent: {response}")
        print()


if __name__ == "__main__":
    main()

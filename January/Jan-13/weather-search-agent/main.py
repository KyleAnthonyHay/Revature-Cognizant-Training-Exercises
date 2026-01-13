"""
Simple LLM Query with AWS Bedrock
"""
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from langchain.agents import create_agent
from tools import get_weather, get_timezone

BEDROCK_MODEL = "bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0"


def main():
    print("Initializing agent...")
    
    agent = create_agent(
        model=BEDROCK_MODEL,
        tools=[get_weather, get_timezone],
        system_prompt="You are a helpful assistant that can check weather and timezone information.",
        name="weather_agent"
    )
    
    print("\nAgent ready! Ask me about weather or timezones.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        print("Agent: ", end="", flush=True)
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_input}]
        })
        
        response = result["messages"][-1].content
        print(response)
        print()


if __name__ == "__main__":
    main()
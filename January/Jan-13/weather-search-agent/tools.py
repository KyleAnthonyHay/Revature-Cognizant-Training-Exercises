"""
Weather and timezone tools for the agent.
"""
from langchain_core.tools import tool


# ========================================================
# Tool 1: Get weather
# ========================================================
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Use this when user asks about weather."""
    return f"Weather in {city}: Sunny, 72°F"


# ========================================================
# Tool 2: Get timezone
# ========================================================
@tool
def get_timezone(city: str) -> str:
    """Get the timezone for a city. Use this when user asks about timezone or time."""
    timezones = {
        "New York": "EST (UTC-5)",
        "London": "GMT (UTC+0)",
        "Tokyo": "JST (UTC+9)"
    }
    return timezones.get(city, f"Unknown timezone for {city}")

# Connecting to LangGraph Studio

## Prerequisites

1. Install LangGraph CLI:
```bash
pip install langgraph-cli
```

2. Make sure you have your `.env` file set up with:
```bash
OPENAI_API_KEY=your-key-here
```

## Steps to Connect

1. **Navigate to the exercise directory:**
```bash
cd January/excercises/week-3/thursday/pair_exercise_01_approval_system
```

2. **Start LangGraph Studio:**
```bash
langgraph dev
```

This will:
- Start a local server (usually at `http://localhost:8123`)
- Open LangGraph Studio in your browser automatically
- Watch for changes to your `graph.py` file

## Using LangGraph Studio

Once Studio opens:

1. **Test Safe Operations (No Approval):**
   - Try: `"List files in /home"`
   - This should execute immediately without approval

2. **Test Risky Operations (Requires Approval):**
   - Try: `"Delete temp.log"`
   - Try: `"Write hello to greeting.txt"`
   - These should pause and show an approval prompt

3. **Approval Flow:**
   - When a risky operation is requested, you'll see an approval dialog
   - Review the operation details
   - Click "Approve" to continue or "Reject" to cancel

## Troubleshooting

- **Port already in use?** Use a different port: `langgraph dev --port 8124`
- **Can't find graph.py?** Make sure you're in the correct directory
- **Agent not loading?** Check that all imports are correct and dependencies are installed

## Alternative: Using LangSmith Tracing

If you want to see traces in LangSmith (not Studio), set these environment variables:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=approval-system
```

Then run your agent normally - traces will appear in LangSmith dashboard.

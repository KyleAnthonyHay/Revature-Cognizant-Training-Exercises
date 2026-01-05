"""
Exercise 02: RAG Analysis - Starter Code

Analyze RAG behavior and document observations.

Prerequisites:
- Completed Exercise 01

This starter provides the RAG system for analysis.
"""

import chromadb

# ============================================================================
# SETUP - Pre-built RAG System for Analysis
# ============================================================================

print("=" * 60)
print("Exercise 02: RAG Analysis")
print("=" * 60)


class MockLLM:
    """Simple mock LLM for testing."""
    def generate(self, prompt):
        if "Context:" in prompt and len(prompt) > 100:
            context_start = prompt.find("Context:") + 8
            context_end = prompt.find("Question:")
            context = prompt[context_start:context_end].strip()
            return f"Based on the context: {context[:200]}..."
        return "I don't have enough information to answer that."


class SimpleRAG:
    """Pre-built RAG system for analysis exercises."""
    
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("rag_analysis")
        self.llm = MockLLM()
    
    def add_knowledge(self, documents, ids=None):
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        self.collection.add(documents=documents, ids=ids)
        print(f"[RAG] Added {len(documents)} documents")
    
    def query(self, question, k=3):
        results = self.collection.query(
            query_texts=[question], n_results=k,
            include=["documents", "distances"]
        )
        
        context_docs = results['documents'][0]
        distances = results['distances'][0]
        
        context = "\n".join(context_docs)
        prompt = f"""You are a helpful assistant. Answer based ONLY on the context.
If context doesn't contain enough info, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""
        
        answer = self.llm.generate(prompt)
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": context_docs,
            "distances": distances,
            "num_sources": len(context_docs)
        }


# Initialize RAG with knowledge base
rag = SimpleRAG()
rag.add_knowledge([
    "Python was created by Guido van Rossum and first released in 1991.",
    "Machine learning enables systems to learn from experience without explicit programming.",
    "Docker is a platform for running applications in containers.",
    "REST is an architectural style using HTTP methods for networked applications.",
    "Neural networks are computing systems inspired by biological brains.",
    "Git is a version control system created by Linus Torvalds in 2005."
])


# ============================================================================
# PART 1: RAG Behavior Testing
# ============================================================================

print("\n" + "=" * 60)
print("Part 1: RAG Behavior Testing")
print("=" * 60)

# Direct Match Questions
print("\n--- DIRECT MATCH ---")
direct_questions = [
    "What year was Python released?",
    "What is REST?"
]

# TODO: Test each question and document results
# for q in direct_questions:
#     result = rag.query(q)
#     print(f"Q: {q}")
#     print(f"A: {result['answer'][:100]}...")
#     print(f"Sources: {result['distances']}")
print("TODO: Test direct match questions")


# Synthesis Required
print("\n--- SYNTHESIS REQUIRED ---")
synthesis_questions = [
    "What are two programming languages mentioned and who created them?"
]
print("TODO: Test synthesis questions")


# No Match
print("\n--- NO MATCH ---")
no_match_questions = [
    "What is the capital of France?",
    "How do I make pizza?"
]
print("TODO: Test no-match questions")


# Ambiguous
print("\n--- AMBIGUOUS ---")
ambiguous_questions = [
    "Tell me about technology",
    "What programming tools should I use?"
]
print("TODO: Test ambiguous questions")


# ============================================================================
# PART 2: Edge Case Analysis
# ============================================================================

print("\n" + "=" * 60)
print("Part 2: Edge Case Analysis")
print("=" * 60)

# TODO 2.1: Test edge cases
print("""
Edge Cases to Test:

1. Empty/Short Queries:
   - "" (empty)
   - "?"
   - "python" (single word)

2. Very Long Query:
   [paste a paragraph]

3. Typos:
   - "waht is mashcine lerning?"
   - "pytohn programming"

4. Opposite Meaning:
   - "What is NOT machine learning?"
""")

print("TODO: Document what happens for each edge case")


# TODO 2.2: Failure Mode Identification
print("""
FAILURE MODE TABLE:
| Failure Mode          | Example Query | What Happened | Possible Fix |
|-----------------------|---------------|---------------|--------------|
| No relevant docs      |               |               |              |
| Wrong docs retrieved  |               |               |              |
| Context too short     |               |               |              |
| Context contradicts   |               |               |              |
""")


# ============================================================================
# PART 3: Improvement Experiments
# ============================================================================

print("\n" + "=" * 60)
print("Part 3: Improvement Experiments")
print("=" * 60)

# TODO 3.1: Test different k values
print("""
K-VALUE EXPERIMENT:
| k value | Pros | Cons | Best For |
|---------|------|------|----------|
| k=1     |      |      |          |
| k=3     |      |      |          |
| k=5     |      |      |          |
| k=10    |      |      |          |
""")


# TODO 3.2: Threshold filtering experiment
def query_with_threshold(rag, question, threshold=0.5):
    """Only use documents above relevance threshold."""
    result = rag.query(question)
    # TODO: Filter out low-relevance documents
    # Hint: Check if distance < threshold
    pass


# TODO 3.3: Document changes
print("""
DOCUMENT CHANGE EXPERIMENTS:
1. Add contradicting info: "Python was created in 2020"
   Result: ___

2. Add duplicate: Same Python doc twice
   Result: ___

3. Remove document: Delete Python doc, ask about Python
   Result: ___
""")


# ============================================================================
# PART 4: Written Analysis
# ============================================================================

print("\n" + "=" * 60)
print("Part 4: Written Analysis")
print("=" * 60)

print("""
RAG STRENGTHS:
(What problems does RAG solve well?)
_______________________________________________
_______________________________________________


RAG LIMITATIONS:
(When does RAG fail or produce poor results?)
_______________________________________________
_______________________________________________


PRODUCTION CONSIDERATIONS:
_______________________________________________
_______________________________________________
""")


print("\n" + "=" * 60)
print("Exercise Complete!")
print("=" * 60)

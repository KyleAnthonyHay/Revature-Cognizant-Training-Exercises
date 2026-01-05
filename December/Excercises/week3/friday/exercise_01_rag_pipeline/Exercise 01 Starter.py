"""
Exercise 01: Build a RAG Pipeline - Starter Code

Build a Retrieval-Augmented Generation system.

Prerequisites:
- pip install chromadb

Hints:
- Reading 03 (rag-basics-introduction.md) explains the pattern
- Demo 02 (demo_02_basic_rag.py) has a complete implementation
"""

import chromadb

# ============================================================================
# SETUP
# ============================================================================

print("=" * 60)
print("Exercise 01: Build a RAG Pipeline")
print("=" * 60)

client = chromadb.Client()


# ============================================================================
# PART 1: Knowledge Base Setup
# ============================================================================

print("\n" + "=" * 60)
print("Part 1: Knowledge Base Setup")
print("=" * 60)

# TODO 1.1: Create collection with cosine distance
# collection = client.create_collection(...)
print("TODO: Create 'rag_knowledge_base' collection")


# TODO 1.2: Add knowledge documents
knowledge_docs = [
    "Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability with its notable use of significant indentation.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
    "Docker is a platform for developing, shipping, and running applications in containers. Containers are lightweight, standalone packages that include everything needed to run software.",
    "REST (Representational State Transfer) is an architectural style for designing networked applications. It relies on stateless, client-server communication using HTTP methods.",
    "Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes (neurons) that process information.",
    "Git is a distributed version control system for tracking changes in source code. It was created by Linus Torvalds in 2005 for Linux kernel development."
]

doc_ids = [f"doc_{i+1}" for i in range(len(knowledge_docs))]

# collection.add(documents=knowledge_docs, ids=doc_ids)
print("TODO: Add knowledge documents")


# ============================================================================
# PART 2: Implement the Retriever
# ============================================================================

print("\n" + "=" * 60)
print("Part 2: Implement the Retriever")
print("=" * 60)

# TODO 2.1: Create retriever function
def retrieve(question, k=3):
    """
    Retrieve top-k relevant documents for a question.
    
    Returns:
        dict with 'documents' and 'distances'
    
    Hint: Use collection.query(query_texts=[question], n_results=k, ...)
    """
    pass


# TODO 2.2: Test the retriever
test_questions = [
    "Who created Python?",
    "What is machine learning?",
    "How does Docker work?"
]
print("TODO: Test retriever with sample questions")


# ============================================================================
# PART 3: Implement the Augmenter
# ============================================================================

print("\n" + "=" * 60)
print("Part 3: Implement the Augmenter")
print("=" * 60)

# TODO 3.1: Create prompt builder
def build_prompt(question, context_docs):
    """
    Build a prompt with the question and retrieved context.
    
    Template:
        You are a helpful assistant. Answer based ONLY on the context.
        If context doesn't contain enough info, say so.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
    """
    # Hint: Join context_docs with newlines
    pass


# TODO 3.2: Test the prompt builder
print("TODO: Print a sample prompt to verify format")


# ============================================================================
# PART 4: Implement the Generator
# ============================================================================

print("\n" + "=" * 60)
print("Part 4: Implement the Generator")
print("=" * 60)

# TODO 4.1: Create Mock LLM
class MockLLM:
    """
    Mock LLM for learning purposes.
    
    In production, this would call OpenAI/Anthropic/etc.
    
    Logic:
    - If prompt has "Context:" with content, generate grounded response
    - Otherwise, say "I don't have enough information"
    
    Hint: Check if "Context:" in prompt and len(prompt) > 100
    """
    def generate(self, prompt):
        # Your implementation here
        pass


# TODO 4.2: Test the generator
print("TODO: Test MockLLM with a sample prompt")


# ============================================================================
# PART 5: Put It All Together
# ============================================================================

print("\n" + "=" * 60)
print("Part 5: Complete RAG System")
print("=" * 60)

# TODO 5.1: Create the RAG class
class SimpleRAG:
    """
    Complete RAG system combining retrieval, augmentation, and generation.
    
    Hint: Demo 02's BasicRAG class has this exact structure
    """
    
    def __init__(self):
        # Initialize Chroma client, collection, and MockLLM
        pass
    
    def add_knowledge(self, documents):
        # Add documents to collection
        pass
    
    def query(self, question, k=3):
        """
        Answer a question using RAG.
        
        Steps:
        1. RETRIEVE - Get relevant docs from vector store
        2. AUGMENT - Build prompt with context
        3. GENERATE - Get answer from LLM
        
        Returns:
            dict with 'question', 'answer', 'sources_used'
        """
        pass


# TODO 5.2: Test the complete system
print("TODO: Create SimpleRAG instance and test with questions")

test_rag_questions = [
    "Who created Python and when?",
    "What are neural networks?",
    "How do I use Git?",
    "What is the best pizza topping?"  # No relevant context
]


# ============================================================================
# PART 6: Enhanced Output
# ============================================================================

print("\n" + "=" * 60)
print("Part 6: Enhanced Output")
print("=" * 60)

# TODO 6.1 & 6.2: Create display function
def display_answer(result, question):
    """
    Pretty print the RAG result.
    
    Format:
    ----------------------------
    Question: "..."
    ----------------------------
    
    Answer: ...
    
    Sources: X documents used
      - "First few words of source..."
    
    ----------------------------
    """
    pass


print("\n" + "=" * 60)
print("Exercise Complete!")
print("=" * 60)

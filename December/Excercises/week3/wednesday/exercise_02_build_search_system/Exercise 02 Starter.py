"""
Exercise 02: Build a Semantic Search System - Starter Code

Build a complete search engine over a document corpus.

Prerequisites:
- pip install chromadb sentence-transformers

Hints:
- Demo 02 shows a complete search pipeline
- Reading 03 covers k-NN search logic
- Reading 04 covers score interpretation
"""

import chromadb
import time

# ============================================================================
# SETUP
# ============================================================================

print("=" * 60)
print("Exercise 02: Build a Semantic Search System")
print("=" * 60)

client = chromadb.Client()


# ============================================================================
# PART 1: Document Ingestion
# ============================================================================

print("\n" + "=" * 60)
print("Part 1: Document Ingestion")
print("=" * 60)

# TODO 1.1: Create collection with Cosine distance
# Hint: metadata={"hnsw:space": "cosine"}
# collection = client.create_collection(...)

print("TODO: Create 'tech_blog' collection with cosine distance")


# TODO 1.2: Add the corpus with metadata
documents = [
    "Introduction to Python programming for beginners",
    "Advanced machine learning techniques using neural networks",
    "How to deploy applications using Docker containers",
    "Building REST APIs with Flask and FastAPI",
    "Understanding data structures: arrays, lists, and trees",
    "Deep learning fundamentals: CNNs and RNNs explained",
    "Kubernetes for container orchestration at scale",
    "Natural language processing with transformers",
    "Database design patterns for scalable applications",
    "Getting started with cloud computing on AWS"
]

ids = [f"doc_{i+1}" for i in range(len(documents))]

# TODO: Define metadata for each document
# Categories: "programming", "ai", "devops", "data"
# Difficulty: "beginner", "intermediate", "advanced"
metadatas = [
    # {"category": "programming", "difficulty": "beginner"},  # doc_1
    # {"category": "ai", "difficulty": "advanced"},           # doc_2
    # ... fill in the rest
]

# collection.add(documents=documents, ids=ids, metadatas=metadatas)
print("TODO: Add documents with metadata")


# ============================================================================
# PART 2: Search Implementation
# ============================================================================

print("\n" + "=" * 60)
print("Part 2: Search Implementation")
print("=" * 60)

# TODO 2.1: Implement basic search function
def search(query, k=5):
    """
    Search for documents similar to the query.
    
    Args:
        query: Natural language search query
        k: Number of results to return
        
    Returns:
        dict with documents, distances, and ids
    """
    # Hint: collection.query(query_texts=[query], n_results=k, include=[...])
    pass


# TODO 2.2: Convert distance to relevance score (0-100)
def distance_to_score(distance):
    """
    Convert Chroma distance to a 0-100 relevance score.
    
    Lower distance -> Higher score
    
    Hint: score = max(0, (1 - distance) * 100)
    """
    pass


# TODO 2.3: Implement filtered search
def search_by_category(query, category, k=3):
    """
    Search within a specific category.
    
    Hint: Add where={"category": category} to query()
    """
    pass


# ============================================================================
# PART 3: Search Quality Testing
# ============================================================================

print("\n" + "=" * 60)
print("Part 3: Search Quality Testing")
print("=" * 60)

# TODO 3.1: Test these queries
test_queries = [
    "How do I start learning to code?",
    "AI and neural networks",
    "deploying apps to production",
    "working with data and databases"
]

print("TODO: Test each query and analyze top 3 results")
# for query in test_queries:
#     results = search(query, k=3)
#     # Analyze and print results


# TODO 3.2: Test edge cases
print("\nEdge Case Tests:")
print("  1. No match: 'Italian cooking recipes' - Results: ___")
print("  2. Exact match: 'Python programming for beginners' - Results: ___")
print("  3. Partial: 'Python' - Results: ___")


# TODO 3.3: Implement threshold filter
def search_with_threshold(query, threshold=50, k=5):
    """
    Only return results above the threshold score.
    
    Hint: Filter out low-scoring results after search
    """
    pass


print("\nThreshold recommendations:")
print("  High-precision (score > ?): Only very relevant")
print("  High-recall (score > ?): Don't miss related docs")


# ============================================================================
# PART 4: Pretty Print Results
# ============================================================================

print("\n" + "=" * 60)
print("Part 4: Pretty Print Results")
print("=" * 60)

# TODO 4.1: Implement display_results function
def display_results(results, query, elapsed_time=0):
    """
    Pretty print search results.
    
    Example output:
    ----------------------------------------
    Search: "query here"
    ----------------------------------------
    
      #1 [Score: 87] Document title
         Category: xxx | Difficulty: xxx
    
    ----------------------------------------
    Found X results in Y seconds
    ----------------------------------------
    """
    pass


print("TODO: Implement pretty print and test with a sample query")


print("\n" + "=" * 60)
print("Exercise Complete!")
print("=" * 60)

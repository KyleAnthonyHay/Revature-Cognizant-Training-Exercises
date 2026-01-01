"""
Exercise 01: Metric Comparison - Starter Code

Compare Euclidean (L2) and Cosine distance metrics.

Prerequisites:
- pip install chromadb sentence-transformers numpy

Hints:
- Reading 02 (distance-metrics-euclidean-cosine.md) has formulas
- Demo 01 shows collection creation with different metrics
"""

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================================
# SETUP
# ============================================================================

print("=" * 60)
print("Exercise 01: Metric Comparison")
print("=" * 60)

client = chromadb.Client()
model = SentenceTransformer('all-MiniLM-L6-v2')


# ============================================================================
# PART 1: Mathematical Understanding (by hand first!)
# ============================================================================

print("\n" + "=" * 60)
print("Part 1: Mathematical Understanding")
print("=" * 60)

# Task 1.1: Calculate by hand for these vectors:
A = np.array([1, 2])
B = np.array([3, 6])  # Same direction as A, different magnitude
C = np.array([2, 1])  # Different direction from A

print("""
VECTORS:
  A = [1, 2]
  B = [3, 6] (same direction as A)
  C = [2, 1] (different direction from A)

YOUR HAND CALCULATIONS:
  Euclidean A→B: _______
  Euclidean A→C: _______
  Cosine similarity A·B: _______
  Cosine similarity A·C: _______

PREDICTIONS:
  Which is "closer" to A using Euclidean? _______
  Which is "closer" to A using Cosine? _______
""")

# TODO: Verify your hand calculations with code
# def euclidean_distance(a, b):
#     ...
# def cosine_similarity(a, b):
#     ...


# ============================================================================
# PART 2: Implementation
# ============================================================================

print("\n" + "=" * 60)
print("Part 2: Create Collections with Different Metrics")
print("=" * 60)

# TODO 2.1: Create two collections with different distance metrics
# Hint: metadata={"hnsw:space": "cosine"} or {"hnsw:space": "l2"}

# cosine_collection = client.create_collection(
#     name="text_cosine",
#     metadata={"hnsw:space": ???}
# )

# l2_collection = client.create_collection(
#     name="text_l2",
#     metadata={"hnsw:space": ???}
# )

print("TODO: Create collections with cosine and l2 metrics")


# TODO 2.2: Add these documents to BOTH collections
test_documents = [
    "Machine learning is transforming the world",
    "MACHINE LEARNING IS TRANSFORMING THE WORLD",  # Same, uppercase
    "Deep learning uses neural networks",
    "I love eating pizza on Friday nights",
    "The weather is sunny and warm today"
]

doc_ids = [f"doc_{i}" for i in range(len(test_documents))]

# Hint: Add same documents to both collections
# cosine_collection.add(documents=test_documents, ids=doc_ids)
# l2_collection.add(documents=test_documents, ids=doc_ids)

print("TODO: Add test documents to both collections")


# TODO 2.3: Query and compare results
print("\n" + "=" * 60)
print("Part 2.3: Query and Compare")
print("=" * 60)

query = "AI and machine learning applications"

# Query both collections
# cosine_results = cosine_collection.query(query_texts=[query], n_results=5)
# l2_results = l2_collection.query(query_texts=[query], n_results=5)

print(f"Query: '{query}'")
print("\nTODO: Compare Cosine vs L2 results")


# ============================================================================
# PART 3: Analysis
# ============================================================================

print("\n" + "=" * 60)
print("Part 3: Analysis")
print("=" * 60)

# TODO 3.1: Fill in your analysis
print("""
COSINE Results:
1. [distance: ____] ________________________
2. [distance: ____] ________________________
3. [distance: ____] ________________________

L2 Results:
1. [distance: ____] ________________________
2. [distance: ____] ________________________
3. [distance: ____] ________________________
""")

# TODO 3.2: Compare doc_1 and doc_2 (case difference)
print("""
CASE SENSITIVITY TEST:
  Doc 1: "Machine learning is transforming the world"
  Doc 2: "MACHINE LEARNING IS TRANSFORMING THE WORLD"
  
  Cosine similarity: ____
  L2 distance: ____
  
  Which handles case differences better? ____
  Why? ____
""")

# TODO 3.3: Fill in the decision guide
print("""
DECISION GUIDE:
| Scenario                        | Metric | Reason |
|---------------------------------|--------|--------|
| Text semantic search            | ???    | ???    |
| Normalized image embeddings     | ???    | ???    |
| Geographic coordinates          | ???    | ???    |
| Vectors where magnitude matters | ???    | ???    |
""")


print("\n" + "=" * 60)
print("Exercise Complete!")
print("=" * 60)

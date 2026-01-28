"""
Women's Review Search Tools - ChromaDB Cloud
"""
import os
import chromadb
from langchain_core.tools import tool

VALID_CATEGORIES = ["Intimates", "Dresses", "Pants", "Knits", "Skirts"]

def get_collection():
    client = chromadb.CloudClient(
        tenant=os.getenv("CHROMA_TENANT", "default_tenant"),
        database=os.getenv("CHROMA_DATABASE", "default_database"),
        api_key=os.getenv("CHROMA_API_KEY")
    )
    return client.get_collection("womens_reviews")


# ========================================================
# Tool 1: Search reviews by category
# ========================================================
@tool
def search_by_category(category: str) -> str:
    """Search reviews by clothing category. Categories: Intimates, Dresses, Pants, Knits, Skirts."""
    if category not in VALID_CATEGORIES:
        return f"Invalid category. Choose from: {VALID_CATEGORIES}"
    
    collection = get_collection()
    results = collection.get(
        where={"category": category},
        include=["documents", "metadatas"]
    )
    
    if not results["ids"]:
        return f"No reviews found for category: {category}"
    
    output = f"Found {len(results['ids'])} reviews for {category}:\n\n"
    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"]), 1):
        output += f"{i}. Rating: {meta.get('rating')}/5 | Age: {meta.get('age')}\n   {doc[:200]}...\n\n"
    return output


# ========================================================
# Tool 2: Semantic search by topic/meaning
# ========================================================
@tool
def search_by_topic(query: str, n_results: int = 5) -> str:
    """Search reviews by topic or meaning. Use this for questions like 'find reviews about comfort' or 'reviews mentioning quality'."""
    collection = get_collection()
    total_count = collection.count()
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    if not results["ids"][0]:
        return f"No reviews found matching: {query}"
    
    output = f"Searched {total_count} total reviews. Top {len(results['ids'][0])} matching '{query}':\n\n"
    for i, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), 1):
        similarity = (1 - dist) * 100
        output += f"{i}. [{meta.get('category')}] Rating: {meta.get('rating')}/5 | Age: {meta.get('age')} | Match: {similarity:.1f}%\n   {doc[:200]}...\n\n"
    return output


# ========================================================
# Tool 3: Combined search (topic + category filter)
# ========================================================
@tool
def search_combined(query: str, category: str, n_results: int = 5) -> str:
    """Search reviews by topic within a specific category. Use when user wants semantic search filtered by category."""
    if category not in VALID_CATEGORIES:
        return f"Invalid category. Choose from: {VALID_CATEGORIES}"
    
    collection = get_collection()
    
    all_ids = collection.get(where={"category": category}, include=[])["ids"]
    total_in_category = len(all_ids)
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"category": category},
        include=["documents", "metadatas", "distances"]
    )
    
    if not results["ids"][0]:
        return f"No reviews found matching '{query}' in {category}"
    
    output = f"Searched {total_in_category} {category} reviews. Top {len(results['ids'][0])} matching '{query}':\n\n"
    for i, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), 1):
        similarity = (1 - dist) * 100
        output += f"{i}. Rating: {meta.get('rating')}/5 | Age: {meta.get('age')} | Match: {similarity:.1f}%\n   {doc[:200]}...\n\n"
    return output


# ========================================================
# Tool 4: Filter by rating
# ========================================================
@tool
def search_by_rating(min_rating: int, category: str = None) -> str:
    """Search reviews with minimum rating (1-5). Optionally filter by category."""
    if min_rating < 1 or min_rating > 5:
        return "Rating must be between 1 and 5"
    
    collection = get_collection()
    where_filter = {"rating": {"$gte": min_rating}}
    if category:
        if category not in VALID_CATEGORIES:
            return f"Invalid category. Choose from: {VALID_CATEGORIES}"
        where_filter = {"$and": [{"rating": {"$gte": min_rating}}, {"category": category}]}
    
    results = collection.get(
        where=where_filter,
        include=["documents", "metadatas"]
    )
    
    if not results["ids"]:
        return f"No reviews found with rating >= {min_rating}"
    
    cat_text = f" in {category}" if category else ""
    output = f"Found {len(results['ids'])} reviews with rating >= {min_rating}{cat_text}:\n\n"
    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"]), 1):
        output += f"{i}. [{meta.get('category')}] Rating: {meta.get('rating')}/5 | Age: {meta.get('age')}\n   {doc[:200]}...\n\n"
    return output

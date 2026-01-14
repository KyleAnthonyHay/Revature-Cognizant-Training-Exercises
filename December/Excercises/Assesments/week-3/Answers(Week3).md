### Q1: What is a vector database and how does it differ from a traditional relational database?

A vector database captures the semantic meaning of data as high-dimensional vectors using embeddings. <br>
A relational database stores data in tables and uses exact matching with SQL queries. Vector databases search by "semantic similarity" and can return multiple results.

### Q2: What is an embedding in the context of AI/ML?

Embedding is the process of representing dat as a high dimentional numerical vector. This allows data with similar semantic meaning to have similar vectors.

### Q3: What are the three main client types in Chroma DB?

- In-memory: Ephemeral, only keeps data while the program is running
- PersistentClient: Data is saved to disk
- HttpClient: this connects to a remote chroma sever over HTTP.

### Q4: What is the difference between Euclidean distance and Cosine similarity?

Euclidean Distance accounts for magnitude of a vector and compares the straight-line distance between two vectors. Cosine similarity accounts for the direction of the vector and its angle with respect to 0.0.

### Q5: What does RAG stand for and what problem does it solve?

Retrieval Augmented Generation. This is an alternative to fine-tuning a model. The benefits are it allows the model to retrieve up-to-date data and context (improving knowledge cutoff) and reduce model hallucination by grounding its response with the relevant data.

### Q6: What is a "collection" in Chroma DB?

A collection is the equivalent of a relational database table but for vector databases. It groups relevant data via embeddings. The actual content lies within the metadata. 

### Q7: What is the relationship between Cosine Similarity and Cosine Distance?

They are inversely preportional: Cosine Distance = 1 - Cosine Similarity. 

### Q8: What is the default embedding model used by Chroma DB?

Chroma uses the sentence transformer: all-MiniLM-L6-v2. It has 384 dimensions which is a good balance between quality and speed.

### Q9: What are the three core components of a RAG system?



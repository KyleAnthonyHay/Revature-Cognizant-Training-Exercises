### Q1: What is TensorBoard and why is it useful for training neural networks?

A visualization tool that helps you see training, metrics, debugging and callbacks

### Q2:What is an autoencoder and what are its main components?

An autoEncoder is a type of Neural network designed to learn how to compress and reconstruct user input. 

Encoder: compressesinput by putting it into a smaller latent space
Bottleneck: :  Compressed prepresentation of user input(the layer that stores the reduced form of the data).
decoder: attempts to reconstructs the original input 

### Q3: What is backpropagation and how does it enable neural networks to learn?

Backpropagation is an algorithm that calculates the gradient of the loss function for each weight in the neural network. 

It does a forward pass through the layers making predictions on the way. It calculates the loss using the loss function at the end. It then uses the chain rule to compute the gradients that reduce the error.

Finally, it updates the weights in that direction.

### Q4:Explain the difference between SGD, Adam, and RMSprop optimizers.

SGD: updates weights using a fixed learning rate
RMSprop (adaptive, good convergence): updates weights individually based on how recent gradients have behaved using average magnitude
Adam (adaptive, best convergence): updates weights individually like RMSprop but also incorporates momentum, which analyzes magnitude and direction


### Q5: What is batch normalization and why is it used?

- It's a method that standardizes the inputs of each layer to reduce internal covariate shift, which helps the network train faster and more stably. It also includes learnable parameters to scale and shift the outputs.

- It normalizes the inputs of each layer to reduce something called internal covariate shift. This basically means it keeps the distribution of inputs consistent, which helps the network train faster and more reliably. It also adds a bit of stability and can let you use a larger learning rate. So in short, it normalizes, speeds up training, and makes things more stable.

### Q5: What does RAG stand for and what problem does it solve?

RAG is retrieval augmented generation. It solves:

Hallucination: Helps the AI model avoid imagining false content by grounding it with relevant data.

Knowledge Cutoff: by allowing the model to retrieve more recent context from up-to-date data sources.

### Q6: What is a "collection" in Chroma DB?

It's the equivalent of a relational database table but for vector databases. It's a group of related embeddings represented by vectors.

Each collection has its own metadata schema.

### Q7: What is the relationship between Cosine Similarity and Cosine Distance?

They are inversely proportional. A cosine similarity of 0.85 equals a cosine distance of 0.15. Cosine Distance = 1 - Cosine Similarity

### Q8: What is the default embedding model used by Chroma DB?

sentence-transformers/all-MiniLM-L6-v2. It has 384-dimensional vectors and is a middle ground between speed and quality.

### Q9: What are the three core components of a RAG system?

- The Document Store which is the vector database. This stores the content as embeddings so you can search for them semantically.

- Retriever: Finds relevant documents from the Vector database from a given query using semantic similarity.

- Generator (LLM): From the context provided by the retriever it generates a response.

The Orchestrator coordinates the flow between each of the 3 steps.

### Q10: What command installs Chroma DB in Python?
pip install chromadb
pip install sentence-transformers - for local embedding support

### Q11: What does the n_results parameter control in a Chroma query?

It determines how many items are returned from a request by implementing k-nearest neighbors. K = the number of results.

### Q12: What are the four CRUD operations in Chroma DB?
Create: collection.add() - Add new documents with embeddings and metadata <br>
Read: collection.get() <br>
Update: collection.update() <br>
collection.query()  <br>
Delete: collection.delete()  <br>

### Q13: What is the purpose of metadata in a vector database?

It allows you to store additional information. You can also filter queries add Additional attributes, structure data.

### Q14: What are common use cases for vector databases?

Vector databases allow for Semantic search(find docs by meaning), Recommendations(like netflix), Image similarity(like googles photo tab), RAG(to Ground LLM responses)

### Q15: Why is Cosine similarity preferred over Euclidean distance for text embeddings?

Cosine similarity accounts for direction, not magnitude. Two documents with similar meaning can have different lengths, so Euclidean distance can result in false negatives.

### Q16: What is the heartbeat() method in Chroma DB used for?

heartbeat() is a method that verifies the health of the Chroma DB client. It returns a timestamp if the connection is successful.

### Q17: What advantages does RAG have over fine-tuning an LLM?

Building a RAG application is cheaper and faster than training a model. You can get up-to-date information through the retrieval step of RAG as well as cite your resources.

## Intermediate (Application) - 25%

### Q18: You're building a semantic search system for customer support FAQs. A user asks "How do I get my money back?" but your database has a document titled "Refund Policy." How would vector search help here, compared to keyword search?

Vector databases store documents by semantic meaning and not by character similarity (via embeddings). The search will associate the document with the request without using keyword matching.


Q19: Your RAG system is returning documents with cosine distances above 0.7 for most queries. Users complain that answers seem irrelevant. What would you do?

I would filter search results to only return values with a cosine distance threshold of 0.4 or less. This is a good middle ground between 'similar' documents and 'potentially relevant' documents. <br>

This will help maintain quality control.

### Q20: When would you choose upsert() over add() in Chroma DB?

- If you don't kow weather or not hte document exists
- to handle erros for fialed uploads
- to simplify code: isntead of implementing an if block that has insert/update logic

### Q21: You need to configure Chroma to use Euclidean distance instead of the default Cosine. How would you do this?

When we create the collection, we assign the metadata tag hnsw:space to l2 (which is Euclidean distance).

### Q22: How would you implement filtered search to find documents in a specific category?

You use a combined query by adding the 'where' metadata tag and add the parameters you want to filter by.

### Q23: What happens if you try to query an empty collection in Chroma?

It gracefully handles the request with no error by returning an empty result structure.

## Advanced (Deep Dive) - 5%

### Q24: Explain what happens "under the hood" when you call collection.add(documents=["Hello world"], ids=["doc1"]) in Chroma without specifying an embedding function.

It uses chroma's default embedding model all-MiniLM-L6-v2, it has 384 dimensions which is a middle gorund between performacnce and quality. it uses HNSW indexing for effecient similairy search. 

### Q25: In a production RAG system processing thousands of queries per minute, what are the key performance considerations and trade-offs?

When handling many queries, it's important to cache frequent requests and use faster models to do the embedding.

If vector search speed is slow, HNSW might be a faster but less accurate solution.

Increasing the chunking size can make retrieval quicker but searches are less precise.

Reducing K-value reduces search time but it reduces the documents received.

Index Updates: Every time a document is added, indexing is updated (which can be expensive). Therefore, it's more efficient to update items in batches than one at a time.
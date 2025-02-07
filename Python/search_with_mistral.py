import ollama
from sentence_transformers import SentenceTransformer
import qdrant_client
import sys
import codecs

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())  # ✅ Force UTF-8 encoding for Windows


# ✅ Load the Sentence Transformer Model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Same model used for embedding

# ✅ Connect to Qdrant (Ensure it's running)
qdrant = qdrant_client.QdrantClient("http://localhost:6333")
collection_name = "text_chunks_embeddings"

def search_qdrant(query_text, top_k=3):
    """Searches Qdrant for similar text chunks."""
    query_vector = model.encode(query_text).tolist()

    results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )

    retrieved_chunks = []
    
    #print("\n🔍 Retrieved Chunks:")
    for hit in results:
        chunk_text = hit.payload["text"]
        retrieved_chunks.append(chunk_text)
        #print(f"- {chunk_text} (Score: {hit.score})\n{'-'*50}")

    return retrieved_chunks

def ask_mistral_with_context(query):
    """Retrieves relevant context from Qdrant and queries Mistral for an improved answer."""
    
    # ✅ Retrieve relevant text chunks
    context_chunks = search_qdrant(query, top_k=2)
    
    # ✅ Format the prompt with retrieved context
    context_text = "\n\n".join(context_chunks)[:1000]
    prompt = f"""Use the following retrieved information as context to answer the question.

    Context:
    {context_text}

    Question: {query}
    Answer:
    """

    # ✅ Query the Mistral model on Ollama
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])

    print("\nMistral's Response:")
    return (response["message"]["content"])

# Example usage
query2 = sys.argv[1]  # ✅ Get query from .NET parameter
result = ask_mistral_with_context(query2)
print(result)

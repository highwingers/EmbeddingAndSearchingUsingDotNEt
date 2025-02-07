import os
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.models import VectorParams, PointStruct
from tqdm import tqdm  # Progress bar for large folders
import tiktoken  # Tokenizer for chunking

# ✅ Load Sentence Transformer Model (Small, Fast & Free)
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dimensional embeddings

# ✅ Connect to Qdrant (Ensure it's running on localhost)
qdrant = qdrant_client.QdrantClient("http://localhost:6333")
collection_name = "text_chunks_embeddings"

# ✅ Create collection if it doesn’t exist
qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance="Cosine")
)

def chunk_text(text, max_tokens=256):
    """Splits text into chunks of max_tokens size using a tokenizer."""
    tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI tokenizer for efficiency
    tokens = tokenizer.encode(text)
    
    # Split into chunks
    chunks = [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    
    # Decode tokens back into text
    return [tokenizer.decode(chunk) for chunk in chunks]

def process_text_files(folder_path):
    """Reads text files, chunks them, generates embeddings, and stores in Qdrant."""
    
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".txt")]
    
    if not file_paths:
        print("❌ No text files found in the folder.")
        return

    chunk_id = 1  # Unique ID for each chunk

    for file_path in tqdm(file_paths, desc="Processing files"):
        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read().strip()

            if not text:
                print(f"⚠ Skipping empty file: {file_path}")
                continue

            # ✅ Chunk the text
            chunks = chunk_text(text, max_tokens=256)

            if not chunks:  # Ensure chunks exist before using
                print(f"⚠ No valid chunks found in {file_path}, skipping...")
                continue

            # ✅ Generate embeddings & store in Qdrant
            for chunk_num, chunk in enumerate(chunks):
                embedding = model.encode(chunk).tolist()

                qdrant.upsert(
                    collection_name=collection_name,
                    points=[
                        PointStruct(
                            id=chunk_id,
                            vector=embedding,
                            payload={
                                "file_name": os.path.basename(file_path),
                                "chunk_num": chunk_num + 1,
                                "text": chunk
                            }
                        )
                    ]
                )
                chunk_id += 1

            print(f"✅ Processed {len(chunks)} chunks for: {file_path}")

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

# Example Usage: Replace with your folder path
process_text_files("Data")  # e.g., "/Users/username/Documents/texts"

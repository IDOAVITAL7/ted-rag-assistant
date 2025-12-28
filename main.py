import pandas as pd
from flask import Flask, request, jsonify
from openai import OpenAI
from pinecone import Pinecone
import pandas as pd
from flask import Flask, request, jsonify
from openai import OpenAI  # We use the standard OpenAI client but with a custom base_url
from pinecone import Pinecone
import os
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

# --- Configuration ---
MY_API_KEY = os.getenv("MY_API_KEY")
MY_BASE_URL = os.getenv("MY_BASE_URL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "ted-talks"
# PINECONE_API_KEY = "pcsk_2vPjW6_2E9GyxZ5421LHe8RcoMHaUZUyvQPXNft4g1mNVqVSgdgM78z7wTZsgo9dJm1XJd"
# MY_API_KEY = "sk-y9YC-G-IETn_6iZXMCtEHA"
# MY_BASE_URL = "https://api.llmod.ai"
# Check if keys are missing (helpful for debugging)
if not MY_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("API Keys are missing! Check your Environment Variables.")

# Initialize clients
# The base_url tells Python to go to your specific platform instead of OpenAI's servers
client = OpenAI(api_key=MY_API_KEY, base_url=MY_BASE_URL)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

RAG_STATS = {
    "chunk_size": 1024,
    "overlap_ratio": 0.3,
    "top_k": 10
}


def get_embedding(text):
    # Using the embedding model from your platform
    response = client.embeddings.create(
        input=text,
        model="RPRTHPB-text-embedding-3-small"  # Or text-embedding-3-large, check your platform's exact name
    )
    return response.data[0].embedding


def upload_data_to_pinecone(csv_path):
    print("Starting data upload to Pinecone...")
    df = pd.read_csv(csv_path).iloc[50:]

    chunk_size = RAG_STATS["chunk_size"]
    overlap_size = int(chunk_size * RAG_STATS["overlap_ratio"])
    step_size = chunk_size - overlap_size

    for _, row in df.iterrows():
        transcript = str(row['transcript'])
        talk_id = str(row['talk_id'])
        title = row['title']

        # Create chunks with overlap
        chunks = []
        for i in range(0, len(transcript), step_size):
            chunk = transcript[i: i + chunk_size]
            chunks.append(chunk)
            if i + chunk_size >= len(transcript):
                break

        # Upload each chunk
        for i, chunk_text in enumerate(chunks):
            vector = get_embedding(chunk_text)
            index.upsert(vectors=[{
                "id": f"{talk_id}_{i}",  # Unique ID for each chunk
                "values": vector,
                "metadata": {
                    "talk_id": talk_id,
                    "title": title,
                    "text": chunk_text
                }
            }])
        print(f"Uploaded: {title} ({len(chunks)} chunks)")
    print("Upload complete!")


@app.route('/api/prompt', methods=['POST'])
@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get("question")

    # 1. Retrieve
    query_vector = get_embedding(question)
    results = index.query(vector=query_vector, top_k=RAG_STATS["top_k"], include_metadata=True)

    # 2. Context building
    context_list = []
    context_text = ""
    for res in results['matches']:
        info = {
            "talk_id": res['metadata'].get('talk_id'),
            "title": res['metadata'].get('title'),
            "chunk": res['metadata'].get('text'),
            "score": res['score']
        }
        context_list.append(info)
        context_text += f"\nTitle: {info['title']}\nContent: {info['chunk']}\n"

    # 3. Generate with gpt-4o-mini
    system_prompt = (
        "You are a TED Talk assistant. Answer the user's question using ONLY the provided context. "
        "If the context contains multiple talks related to the topic, list them as requested. "
        "If the answer is absolutely not in the context, say 'I don't know'."
    )
    user_prompt = f"Context: {context_text}\n\nQuestion: {question}"

    completion = client.chat.completions.create(
        model="RPRTHPB-gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return jsonify({
        "response": completion.choices[0].message.content,
        "context": context_list,
        "Augmented_prompt": {
            "System": system_prompt,
            "User": user_prompt
        }
    })


@app.route('/api/stats', methods=['GET'])
def stats():
    return jsonify(RAG_STATS)

@app.route('/')
def home():
    return "TED Talk RAG Assistant API is live! Use /api/stats or /api/ask."
    
if __name__ == '__main__':
    # Uncomment the next line ONLY for the first run to upload data:
    # upload_data_to_pinecone("ted_talks_en.csv")

    app.run(port=5000,debug=True)




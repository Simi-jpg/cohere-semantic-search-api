import os
import cohere
import numpy as np

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://cohere-search-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# List of sentences to be embedded for semantic search and recommendations
def load_documents():
    with open("documents.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]
    
    
documents = load_documents()



class DocumentRequest(BaseModel):
    text: str

# Embed the documents using Cohere's embedding model (numerical representations of meaning)
response = co.embed(
    texts=documents,
    model="embed-english-v3.0", # ML model
    input_type="search_document" # stored data
)

# convert the list of embeddings to a NumPy array for efficient similarity calculations
documents_embeddings = np.array(response.embeddings)


def search_documents(query): # user query
    # Convert query to embedding
    query_response = co.embed(
        texts=[query], # single query, so we wrap it in a list
        model="embed-english-v3.0",
        input_type="search_query"
    )

    # get the embedding vector for the query
    query_embeddings = np.array(query_response.embeddings[0]) # getting 0 index because we are embedding a single query

    # compare with all document embeddings (get dot product)
    similarities = np.dot(documents_embeddings, query_embeddings)

    # get the index of the highest similarity score
    top_match_idx = np.argsort(similarities)[-3:][::-1] # get top 3 matches, sorted in descending order

    # return the best matching document
    return [documents[i] for i in top_match_idx]


# API endpoint to handle search requests
@app.get("/search") # URL endpoint for search
def search(query: str): # query parameter from the URL
    result = search_documents(query)
    return {"query": query, "best_match": result}


@app.post("/documents") # URL endpoint for adding documents
def add(document: DocumentRequest):
    global documents_embeddings

    # Add the new document to the list
    documents.append(document.text)
    with open("documents.txt", "a") as f:
        f.write(document.text + "\n")

    # Convert to an embedding
    new_embedding_response = co.embed(
        texts=[document.text],
        model="embed-english-v3.0",
        input_type="search_document"
    )

    # convert into np array
    new_embedding = np.array(new_embedding_response.embeddings)

    # add to the embedding matrix
    documents_embeddings = np.vstack([documents_embeddings, new_embedding])

    return {
        "message": "Document added successfully.",
        "text": document.text
    }


# test
# print("Documents:", len(documents))
# print("Embeddings shape:", documents_embeddings.shape)
# print(documents_embeddings[0])  # print the first embedding to verify


# result = search_documents("How do teams ship code faster?")
# print("Best match:", result)
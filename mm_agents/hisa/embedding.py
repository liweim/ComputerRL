#!/usr/bin/env python3
"""
BGE Embedding Service
Provides REST API for text embedding using BGE model
"""
import os
# Limit tokenizer parallelism to avoid resource exhaustion
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAYON_NUM_THREADS"] = "4"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
from FlagEmbedding import BGEM3FlagModel
import numpy as np
import uvicorn
import logging
import requests

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="BGE Embedding Service", version="1.0.0")

# Global model instance
model = None

class EmbeddingRequest(BaseModel):
    """Request model for embedding"""
    texts: Union[str, List[str]]
    batch_size: int = 32
    max_length: int = 512

class EmbeddingResponse(BaseModel):
    """Response model for embedding"""
    embeddings: List[List[float]]
    dimension: int

@app.on_event("startup")
async def load_model():
    """Load BGE model on startup"""
    global model
    try:
        model_path = 'BAAI/bge-m3'
        logger.info(f"Loading BGE model from {model_path}...")
        # Use single device and limit processes to avoid resource exhaustion
        model = BGEM3FlagModel(
            model_path,
            use_fp16=True,
            devices=['cuda:0']  # Use single GPU, or ['cpu'] if no GPU
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "running", "service": "BGE Embedding Service"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "BGE-M3"}

@app.post("/embed", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest):
    """
    Generate embeddings for input text(s)

    Args:
        request: EmbeddingRequest containing texts and optional parameters

    Returns:
        EmbeddingResponse with embeddings and dimension
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Handle single string or list of strings
        texts = [request.texts] if isinstance(request.texts, str) else request.texts

        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided")

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} texts")
        # Only return dense vectors to reduce computation
        embeddings = model.encode(
            texts,
            batch_size=request.batch_size,
            max_length=request.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )['dense_vecs']

        # Convert to list and ensure float32 precision
        embeddings_list = embeddings.astype(np.float32).tolist()

        logger.info(f"Generated embeddings with dimension {len(embeddings_list[0])}")

        return EmbeddingResponse(
            embeddings=embeddings_list,
            dimension=len(embeddings_list[0])
        )

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

class EmbeddingClient:
    """
    Client for calling BGE embedding service via REST API
    """
    def __init__(self, service_url: str = "http://localhost:8000"):
        """
        Initialize embedding client

        Args:
            service_url: URL of the embedding service (default: http://localhost:8000)
        """
        self.service_url = service_url.rstrip('/')
        self._check_health()

    def _check_health(self):
        """Check if the service is healthy"""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"Connected to embedding service at {self.service_url}")
            else:
                print(f"Warning: Embedding service returned status {response.status_code}")
        except Exception as e:
            print(f"Warning: Cannot connect to embedding service at {self.service_url}: {e}")

    def __call__(self, texts: Union[str, List[str]], batch_size: int = 32, max_length: int = 512) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s)

        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing
            max_length: Maximum sequence length

        Returns:
            Single embedding (if input is string) or list of embeddings (if input is list)
        """
        is_single = isinstance(texts, str)

        # Prepare request
        payload = {
            "texts": texts,
            "batch_size": batch_size,
            "max_length": max_length
        }

        try:
            response = requests.post(
                f"{self.service_url}/embed",
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            embeddings = result["embeddings"]

            # Return single embedding if input was single text
            return embeddings[0] if is_single else embeddings

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get embeddings from service: {e}")

def test():
    model = EmbeddingClient('http://localhost:8000')
    sentences_1 = ["What is BGE M3?", "Defination of BM25"]
    sentences_2 = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

    embeddings_1 = model(sentences_1)
    embeddings_2 = model(sentences_2)
    embeddings_1 = np.array(embeddings_1)
    embeddings_2 = np.array(embeddings_2)
    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)
    # [[0.6265, 0.3477], [0.3499, 0.678 ]]
    
if __name__ == "__main__":
    # nohup python embedding_service.py > embedding_service.log 2>&1 &
    uvicorn.run(
        "embedding_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

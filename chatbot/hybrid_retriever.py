"""
Hybrid retriever combining BM25 (keyword) and dense (semantic) search
"""
import numpy as np
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re

class HybridRetriever:
    def __init__(self, E: np.ndarray, metas: List[Dict[str, Any]], 
                 embed_model: SentenceTransformer, alpha: float = 0.5):
        """
        Args:
            E: Normalized embeddings (N x D)
            metas: Metadata for each embedding
            embed_model: SentenceTransformer model
            alpha: Weight for dense retrieval (0-1). 
                   1.0 = only dense, 0.0 = only BM25, 0.5 = balanced
        """
        self.E = E
        self.metas = metas
        self.embed_model = embed_model
        self.alpha = alpha
        
        print("Building BM25 index...")
        tokenized_docs = [self._tokenize(m['text']) for m in metas]
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"BM25 index built with {len(tokenized_docs)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple Hebrew-aware tokenization"""
        # Remove punctuation, split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        # Keep tokens with at least 2 chars
        return [t for t in tokens if len(t) >= 2]
    
    def retrieve(self, query: str, topk: int = 6, 
                 max_per_url: int = 2) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Hybrid retrieval combining BM25 and dense search
        
        Returns:
            List of (combined_score, metadata) tuples
        """
        # 1. Dense retrieval (semantic)
        q_emb = self.embed_model.encode(
            [f"query: {query}"], 
            normalize_embeddings=True
        )[0].astype(np.float32)
        dense_scores = self.E @ q_emb
        
        # 2. BM25 retrieval (keyword)
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 3. Normalize scores to [0, 1]
        def normalize(scores):
            min_s, max_s = scores.min(), scores.max()
            if max_s - min_s < 1e-8:
                return np.zeros_like(scores)
            return (scores - min_s) / (max_s - min_s)
        
        dense_norm = normalize(dense_scores)
        bm25_norm = normalize(bm25_scores)
        
        # 4. Combine scores
        combined = self.alpha * dense_norm + (1 - self.alpha) * bm25_norm
        
        # 5. Get top-k with URL deduplication
        k = min(topk * 4, len(combined))
        top_idx = np.argpartition(-combined, k - 1)[:k]
        top_idx = top_idx[np.argsort(-combined[top_idx])]
        
        picked = []
        per_url = {}
        
        for i in top_idx:
            m = self.metas[i]
            url = m.get("url", "")
            if not url:
                continue
            
            per_url.setdefault(url, 0)
            if per_url[url] >= max_per_url:
                continue
            
            picked.append((float(combined[i]), m))
            per_url[url] += 1
            
            if len(picked) >= topk:
                break
        
        return picked
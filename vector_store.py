import os
import json
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

EMBED_MODEL = "all-MiniLM-L6-v2"


class SimpleFAISSStore:
    def __init__(self, dim=384, index_path="rag/faiss.index", meta_path="rag/meta.json"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.model = SentenceTransformer(EMBED_MODEL)

        if Path(index_path).exists() and Path(meta_path).exists():
            self.index = faiss.read_index(index_path)
            self.meta = json.load(open(meta_path, "r"))
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.meta = []

    def _embed(self, texts: List[str]):
        vectors = self.model.encode(texts, convert_to_numpy=True)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        return vectors.astype("float32")

    def ingest(self, docs: List[Tuple[str, str]]):
        CHUNK = 350
        OVERLAP = 70

        all_chunks = []
        all_meta = []

        for doc_id, text in docs:
            t = text.replace("\n", " ")
            i = 0
            while i < len(t):
                chunk = t[i:i + CHUNK]
                all_chunks.append(chunk)
                all_meta.append({"doc_id": doc_id, "text": chunk})
                i += CHUNK - OVERLAP

        vecs = self._embed(all_chunks)
        self.index.add(vecs)
        self.meta.extend(all_meta)

        faiss.write_index(self.index, self.index_path)
        json.dump(self.meta, open(self.meta_path, "w"), indent=2)

    def search(self, query: str, top_k=5):
        q = self._embed([query])
        D, I = self.index.search(q, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            results.append({"score": float(score), "meta": self.meta[idx]})
        return results


def ingest_folder(store: SimpleFAISSStore, folder="data/kb"):
    docs = []
    for p in Path(folder).glob("*.txt"):
        docs.append((p.name, open(p, "r", encoding="utf-8").read()))
    if docs:
        store.ingest(docs)

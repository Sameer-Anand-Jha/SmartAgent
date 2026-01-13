from typing import List
from .vector_store import SimpleFAISSStore


class RetrievalAgent:
    def __init__(self, store: SimpleFAISSStore, top_k=3):
        self.store = store
        self.top_k = top_k

    def retrieve(self, question: str):
        return self.store.search(question, self.top_k)

    def build_prompt(self, question: str, chunks: List[dict]):
        context = "\n\n".join(
            f"[{c['meta']['doc_id']} | score={c['score']:.3f}] {c['meta']['text']}"
            for c in chunks
        )
        return f"""
You must answer using the retrieved context only.
If answer is not found, say 'I don't know'.

Context:
{context}

Question: {question}

Answer:
""".strip()

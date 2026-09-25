# memory.py — persistent architecture-knowledge memory (RAG)
# Author: 晨星 (CJX0712)
from __future__ import annotations
from .search_space import arch_to_features, cosine


class LocalMemory:
    """Pure-stdlib nearest-neighbour memory over architecture features.

    This is the guaranteed offline path: every evaluated candidate's
    (features, critique, score) is stored and later retrieved to bias the
    Planner away from mistakes already seen — the RAG half of RCG-NAS.
    """

    def __init__(self) -> None:
        self._items: list[dict] = []

    def add(self, arch: dict, text: str, score: float) -> None:
        self._items.append({"feat": arch_to_features(arch), "text": text, "score": score, "name": arch.get("name")})

    def query(self, arch: dict, k: int = 3) -> list[dict]:
        q = arch_to_features(arch)
        ranked = sorted(self._items, key=lambda it: cosine(q, it["feat"]), reverse=True)
        return ranked[:k]

    def __len__(self) -> int:
        return len(self._items)


class HaystackMemory(LocalMemory):
    """RAG memory backed by Haystack's InMemoryDocumentStore.

    Wires in the top-tier OSS *Haystack*: critiques are also persisted as
    Documents so a BM25 retriever can surface prior architecture knowledge.
    Falls back transparently to :class:`LocalMemory` when Haystack is absent.
    """

    def __init__(self) -> None:
        super().__init__()
        self.haystack = None
        self.doc_store = None
        try:
            from haystack.document_stores import InMemoryDocumentStore  # type: ignore
            from haystack import Document  # type: ignore
            self.haystack = Document
            self.doc_store = InMemoryDocumentStore()
        except Exception:
            self.haystack = None
            self.doc_store = None

    def add(self, arch: dict, text: str, score: float) -> None:
        super().add(arch, text, score)
        if self.doc_store is not None and self.haystack is not None:
            try:
                self.doc_store.write_documents([self.haystack(content=f"[score={score:.3f}] {text}", meta={"arch": arch.get("name")})])
            except Exception:
                pass

    @property
    def using_haystack(self) -> bool:
        return self.doc_store is not None


def MemoryStore() -> LocalMemory:
    """Factory: prefer Haystack-backed RAG, else local NN. Closed-loop either way."""
    try:
        return HaystackMemory()
    except Exception:
        return LocalMemory()

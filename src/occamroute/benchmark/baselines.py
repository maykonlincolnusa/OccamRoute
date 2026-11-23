"""Baseline context retrieval pipelines (Brute Force & Naive Chunked RAG) for OccamRoute benchmarks."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from occamroute.graph_builder import RepoGraph, count_tokens

RagBackend = Literal["auto", "tfidf", "faiss"]


@dataclass(frozen=True)
class BaselineContextResult:
    """Retrieved context bundle produced by a baseline pipeline."""

    pipeline: str
    context_text: str
    retrieved_files: list[str]
    retrieved_symbols: list[str]
    context_tokens: int
    prep_latency_ms: float
    num_chunks: int = 0
    retrieval_backend: str = "deterministic"


class BruteForceBaseline:
    """Baseline A: Inject complete source modules into the prompt without structural pruning."""

    def __init__(self, repo_graph: RepoGraph) -> None:
        self.repo_graph = repo_graph

    def retrieve(
        self,
        query: str,
        target_files: list[str] | None = None,
        max_files: int | None = None,
    ) -> BaselineContextResult:
        del query  # Brute Force ignores query specificity and reads whole modules
        t0 = time.perf_counter()

        candidate_files = (
            [f for f in target_files if f in self.repo_graph.file_to_nodes]
            if target_files
            else sorted(self.repo_graph.file_to_nodes.keys())
        )
        if not candidate_files and self.repo_graph.file_to_nodes:
            candidate_files = sorted(self.repo_graph.file_to_nodes.keys())
        if max_files is not None and max_files > 0:
            candidate_files = candidate_files[:max_files]

        blocks: list[str] = []
        retrieved_symbols: list[str] = []
        for rel in candidate_files:
            file_path = self.repo_graph.root_dir / rel
            if file_path.exists():
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            else:
                text = "\n\n".join(
                    self.repo_graph.nodes[nid].source
                    for nid in self.repo_graph.file_to_nodes.get(rel, [])
                    if nid in self.repo_graph.leaf_ids
                )
            blocks.append(f"# FILE: {rel}\n{text}")
            retrieved_symbols.extend(
                nid
                for nid in self.repo_graph.file_to_nodes.get(rel, [])
                if nid in self.repo_graph.leaf_ids
            )

        context_text = "\n\n".join(blocks)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return BaselineContextResult(
            pipeline="Brute Force",
            context_text=context_text,
            retrieved_files=list(candidate_files),
            retrieved_symbols=retrieved_symbols,
            context_tokens=count_tokens(context_text),
            prep_latency_ms=latency_ms,
            num_chunks=len(candidate_files),
            retrieval_backend="full_file",
        )


class NaiveRAGBaseline:
    """Baseline B: Fixed-token overlapping window chunking (512t) with FAISS+BGE or TF-IDF cosine retrieval."""

    def __init__(
        self,
        repo_graph: RepoGraph,
        chunk_tokens: int = 512,
        overlap_tokens: int = 64,
        backend: RagBackend = "auto",
        embed_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.repo_graph = repo_graph
        self.chunk_tokens = max(64, chunk_tokens)
        self.overlap_tokens = max(0, min(overlap_tokens, self.chunk_tokens // 2))
        self.embed_model = embed_model
        self.chunks: list[tuple[str, int, int, str]] = self._build_chunks()
        self.active_backend: Literal["faiss", "tfidf"] = self._select_backend(backend)
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix: Any = None
        self._embedder: Any = None
        self._faiss_index: Any = None
        self._build_index()

    def _select_backend(self, requested: RagBackend) -> Literal["faiss", "tfidf"]:
        if requested == "tfidf":
            return "tfidf"
        try:
            import faiss  # noqa: F401
            from sentence_transformers import SentenceTransformer  # noqa: F401

            return "faiss"
        except ImportError:
            if requested == "faiss":
                raise
            return "tfidf"

    def _build_chunks(self) -> list[tuple[str, int, int, str]]:
        chunks: list[tuple[str, int, int, str]] = []
        for rel in sorted(self.repo_graph.file_to_nodes):
            file_path = self.repo_graph.root_dir / rel
            if not file_path.exists():
                continue
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if not lines:
                continue
            line_toks = [max(1, count_tokens(line)) for line in lines]
            i = 0
            while i < len(lines):
                acc, j = 0, i
                while j < len(lines) and (acc + line_toks[j] <= self.chunk_tokens or j == i):
                    acc += line_toks[j]
                    j += 1
                chunks.append((rel, i + 1, j, "\n".join(lines[i:j])))
                if j >= len(lines):
                    break
                back, next_i = 0, j
                while next_i > i + 1 and back < self.overlap_tokens:
                    next_i -= 1
                    back += line_toks[next_i]
                i = max(i + 1, next_i)
        return chunks

    def _build_index(self) -> None:
        docs = [f"{rel}\n{text}" for rel, _, _, text in self.chunks]
        if not docs:
            return
        if self.active_backend == "faiss":
            import faiss
            from sentence_transformers import SentenceTransformer

            device = "cpu"
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
            except Exception:
                device = "cpu"
            self._embedder = SentenceTransformer(self.embed_model, device=device)
            embs = np.asarray(
                self._embedder.encode(docs, normalize_embeddings=True, show_progress_bar=False),
                dtype="float32",
            )
            self._faiss_index = faiss.IndexFlatIP(embs.shape[1])
            self._faiss_index.add(embs)
        else:
            self._vectorizer = TfidfVectorizer(
                lowercase=True,
                sublinear_tf=True,
                ngram_range=(1, 2),
                token_pattern=r"(?u)\b[A-Za-z_][A-Za-z0-9_]+\b",
            )
            self._tfidf_matrix = self._vectorizer.fit_transform(docs)

    def retrieve(self, query: str, top_k: int = 5) -> BaselineContextResult:
        t0 = time.perf_counter()
        if not self.chunks:
            return BaselineContextResult(
                pipeline="Naive RAG",
                context_text="",
                retrieved_files=[],
                retrieved_symbols=[],
                context_tokens=0,
                prep_latency_ms=(time.perf_counter() - t0) * 1000.0,
                num_chunks=0,
                retrieval_backend=self.active_backend,
            )

        k = min(max(1, top_k), len(self.chunks))
        if self.active_backend == "faiss" and self._faiss_index is not None:
            q_emb = np.asarray(
                self._embedder.encode([query], normalize_embeddings=True, show_progress_bar=False),
                dtype="float32",
            )
            _, indices = self._faiss_index.search(q_emb, k)
            ranked = [int(i) for i in indices[0] if i >= 0]
        else:
            q_vec = self._vectorizer.transform([query])
            sims = linear_kernel(q_vec, self._tfidf_matrix).ravel()
            ranked = [int(i) for i in np.argsort(-sims)[:k]]

        blocks: list[str] = []
        retrieved_symbols: list[str] = []
        retrieved_files: set[str] = set()
        for idx in ranked:
            rel, s_ln, e_ln, txt = self.chunks[idx]
            blocks.append(f"# CHUNK: {rel} (L{s_ln}-L{e_ln})\n{txt}")
            retrieved_files.add(rel)
            for nid in self.repo_graph.file_to_nodes.get(rel, []):
                if nid not in self.repo_graph.leaf_ids:
                    continue
                nd = self.repo_graph.nodes[nid]
                overlap = min(e_ln, nd.end_line) - max(s_ln, nd.start_line) + 1
                span = max(1, nd.end_line - nd.start_line + 1)
                if overlap > 0 and (overlap / span >= 0.35 or (s_ln <= nd.start_line <= e_ln and overlap >= 3)):
                    if nid not in retrieved_symbols:
                        retrieved_symbols.append(nid)

        context_text = "\n\n".join(blocks)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return BaselineContextResult(
            pipeline="Naive RAG",
            context_text=context_text,
            retrieved_files=sorted(retrieved_files),
            retrieved_symbols=retrieved_symbols,
            context_tokens=count_tokens(context_text),
            prep_latency_ms=latency_ms,
            num_chunks=len(ranked),
            retrieval_backend=self.active_backend,
        )

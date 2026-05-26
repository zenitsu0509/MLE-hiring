"""
retriever.py — BM25 corpus retriever with domain pre-filtering and overlapping chunks.

Indexes all .md, .txt, and .json files under data/.
Supports domain-specific search (claude, devplatform, visa) with fallback
to full corpus when domain-specific results are insufficient.
"""

import os
import glob
import re
from typing import List, Tuple, Optional, Set, Dict
from rank_bm25 import BM25Okapi

from config import (
    DATA_DIR, TOP_K_RETRIEVAL, CHUNK_SIZE_WORDS,
    CHUNK_OVERLAP_WORDS, MIN_BM25_SCORE, INDEXABLE_EXTENSIONS,
    COMPANY_TO_DOMAIN,
)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer, lowercase."""
    return re.findall(r'\w+', text.lower())


def _chunk_text(content: str, filepath: str,
                chunk_size: int = CHUNK_SIZE_WORDS,
                overlap: int = CHUNK_OVERLAP_WORDS) -> List[dict]:
    """
    Split content into overlapping chunks of ~chunk_size words.
    Each chunk retains its source filepath.
    """
    words = content.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        # Include filepath in tokens for better matching
        filepath_tokens = _tokenize(filepath)
        chunk_tokens = _tokenize(chunk_text)

        chunks.append({
            "filepath": filepath,
            "content": chunk_text,
            "tokens": filepath_tokens + chunk_tokens,
            "start_word": start,
            "end_word": min(end, len(words)),
        })

        # Move forward by (chunk_size - overlap)
        step = max(chunk_size - overlap, 1)
        start += step
        if start >= len(words) and end < len(words):
            # Edge case: ensure we don't skip the tail
            break

    return chunks


class CorpusRetriever:
    """
    BM25-based retriever over the support corpus.
    
    Features:
        - Indexes all .md, .txt files under data/
        - Overlapping 400-word chunks
        - Domain pre-filtering for faster, more precise retrieval
        - Score normalization to [0, 1]
        - Valid path tracking (only real, existing files)
    """

    def __init__(self, data_dir: str = DATA_DIR):
        self.docs: List[dict] = []          # All chunks: {filepath, content, tokens, ...}
        self.domain_docs: Dict[str, List[dict]] = {}  # domain -> [chunks]
        self._valid_paths: Set[str] = set()

        self._index_corpus(data_dir)

        # Build BM25 indexes
        if self.docs:
            self.bm25_all = BM25Okapi([d["tokens"] for d in self.docs])
        else:
            self.bm25_all = None

        self.bm25_domain: Dict[str, BM25Okapi] = {}
        for domain, docs in self.domain_docs.items():
            if docs:
                self.bm25_domain[domain] = BM25Okapi([d["tokens"] for d in docs])

        print(f"[Retriever] Indexed {len(self.docs)} chunks from "
              f"{len(self._valid_paths)} files across "
              f"{len(self.domain_docs)} domains")

    def _index_corpus(self, data_dir: str):
        """Recursively index all supported files under data_dir."""
        for path in sorted(glob.glob(os.path.join(data_dir, "**", "*"), recursive=True)):
            if not os.path.isfile(path):
                continue

            ext = os.path.splitext(path)[1].lower()
            if ext not in INDEXABLE_EXTENSIONS:
                continue

            # Skip api_specs — not corpus documents
            if "api_specs" in path:
                continue

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue

            if not content.strip():
                continue

            # Normalize path to use forward slashes, relative to repo root
            rel_path = os.path.relpath(path, os.path.dirname(data_dir))
            rel_path = rel_path.replace("\\", "/")

            self._valid_paths.add(rel_path)

            # Chunk the content
            chunks = _chunk_text(content, rel_path)
            self.docs.extend(chunks)

            # Assign to domain
            domain = self._detect_domain(rel_path)
            if domain:
                if domain not in self.domain_docs:
                    self.domain_docs[domain] = []
                self.domain_docs[domain].extend(chunks)

    def _detect_domain(self, filepath: str) -> Optional[str]:
        """Determine which product domain a file belongs to based on path."""
        fp_lower = filepath.lower()
        if "devplatform" in fp_lower:
            return "devplatform"
        elif "claude" in fp_lower:
            return "claude"
        elif "visa" in fp_lower:
            return "visa"
        return None

    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL,
               domain: Optional[str] = None) -> List[Tuple[dict, float]]:
        """
        Search the corpus for relevant chunks.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            domain: Optional domain filter ('claude', 'devplatform', 'visa')
            
        Returns:
            List of (chunk_dict, normalized_score) tuples, sorted by score desc.
        """
        tokens = _tokenize(query)
        if not tokens:
            return []

        results = []

        # 1. Try domain-specific search first
        if domain and domain in self.bm25_domain:
            domain_bm25 = self.bm25_domain[domain]
            domain_docs = self.domain_docs[domain]
            scores = domain_bm25.get_scores(tokens)
            
            max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0
            
            for idx, score in enumerate(scores):
                if score > MIN_BM25_SCORE:
                    norm_score = score / max_score if max_score > 0 else 0
                    results.append((domain_docs[idx], norm_score))

        # 2. Always also search full corpus and merge
        if self.bm25_all is not None:
            all_scores = self.bm25_all.get_scores(tokens)
            max_all_score = max(all_scores) if len(all_scores) > 0 and max(all_scores) > 0 else 1.0
            
            seen_filepaths = {r[0]["filepath"] for r in results}
            
            for idx, score in enumerate(all_scores):
                if score > MIN_BM25_SCORE:
                    doc = self.docs[idx]
                    # Avoid duplicate chunks from domain search
                    chunk_key = (doc["filepath"], doc.get("start_word", 0))
                    if doc["filepath"] not in seen_filepaths or not domain:
                        norm_score = score / max_all_score if max_all_score > 0 else 0
                        # Slight boost for domain-matching results even from full index
                        if domain and self._detect_domain(doc["filepath"]) == domain:
                            norm_score = min(norm_score * 1.1, 1.0)
                        results.append((doc, norm_score))

        # Deduplicate by (filepath, start_word), keep highest score
        deduped = {}
        for doc, score in results:
            key = (doc["filepath"], doc.get("start_word", 0))
            if key not in deduped or deduped[key][1] < score:
                deduped[key] = (doc, score)
        
        results = list(deduped.values())

        # Sort by score descending, take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_valid_paths(self) -> Set[str]:
        """Return all real file paths in the indexed corpus."""
        return self._valid_paths.copy()

    def format_context(self, results: List[Tuple[dict, float]]) -> str:
        """Format retrieved chunks into a context string for the LLM prompt."""
        if not results:
            return "No relevant corpus documents found."

        parts = []
        for i, (doc, score) in enumerate(results, 1):
            parts.append(
                f"--- Document {i} (source: {doc['filepath']}, relevance: {score:.2f}) ---\n"
                f"{doc['content'][:2000]}"  # Cap individual chunk display
            )
        return "\n\n".join(parts)

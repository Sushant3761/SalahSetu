"""
SalahSetu - BNS Legal Retrieval Quality & Reranking Pipeline (Stage 5)

This script enhances the FAISS semantic retrieval pipeline by applying a deterministic,
transparent, multi-signal reranker that combines:
1. FAISS Cosine Similarity score
2. Exact/partial keyword overlap with section title & chunk text
3. Exact phrase matching
4. Section-level awareness

Output Artifact:
- data/embeddings/BNS_reranking_validation_report.json
"""

import json
import hashlib
import re
import math
import argparse
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Tuple

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_FAISS_WEIGHT = 0.60
DEFAULT_KEYWORD_WEIGHT = 0.30
DEFAULT_PHRASE_WEIGHT = 0.10
DEFAULT_CANDIDATE_K = 15

STOPWORDS = {
    "what", "is", "the", "for", "under", "bharatiya", "nyaya", "sanhita",
    "does", "law", "say", "about", "a", "an", "in", "of", "to", "and", "or",
    "by", "with", "from", "at", "it", "this", "that", "be", "are", "on"
}


class BNSRerankedRetriever:
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        chunks_path: Path,
        model_name: str = DEFAULT_MODEL_NAME,
        weight_faiss: float = DEFAULT_FAISS_WEIGHT,
        weight_keyword: float = DEFAULT_KEYWORD_WEIGHT,
        weight_phrase: float = DEFAULT_PHRASE_WEIGHT
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.chunks_path = chunks_path
        self.model_name = model_name

        self.weight_faiss = weight_faiss
        self.weight_keyword = weight_keyword
        self.weight_phrase = weight_phrase

        self._model = None
        self._index = None
        self._metadata = None
        self._chunks_by_id = None

        self._load_artifacts()

    def _load_artifacts(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {self.chunks_path}")

        self._index = faiss.read_index(str(self.index_path))

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            self._chunks_by_id = {c["chunk_id"]: c for c in chunks}

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _extract_keywords_and_phrases(self, query: str) -> Tuple[List[str], str]:
        """
        Extracts non-stopword query keywords and normalized query phrase.
        """
        clean_words = [w.lower() for w in re.findall(r'\w+', query) if w.lower() not in STOPWORDS]
        phrase = " ".join(clean_words)
        return clean_words, phrase

    def compute_reranking_signals(
        self,
        query_words: List[str],
        query_phrase: str,
        section_title: str,
        chunk_text: str
    ) -> Tuple[float, bool]:
        """
        Computes keyword overlap score and exact phrase match flag.
        """
        if not query_words:
            return 0.0, False

        title_lower = section_title.lower()
        text_lower = chunk_text.lower()

        title_hits = sum(1 for w in query_words if w in title_lower)
        text_hits = sum(1 for w in query_words if w in text_lower)

        title_ratio = title_hits / len(query_words)
        text_ratio = text_hits / len(query_words)

        keyword_score = 0.60 * title_ratio + 0.40 * text_ratio

        phrase_match = False
        if query_phrase and (query_phrase in title_lower or query_phrase in text_lower):
            phrase_match = True

        return keyword_score, phrase_match

    def search_faiss_only(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Returns initial top-k FAISS results.
        """
        model = self._get_model()
        query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

        scores, indices = self._index.search(query_emb, top_k)

        results = []
        for rank_idx, (score, vec_idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if vec_idx < 0 or vec_idx >= len(self._metadata):
                continue
            meta = self._metadata[vec_idx]
            cid = meta["chunk_id"]
            chunk_obj = self._chunks_by_id.get(cid, {})

            results.append({
                "rank": rank_idx,
                "score": float(score),
                "chunk_id": cid,
                "section_number": meta["section_number"],
                "section_title": meta["section_title"],
                "chunk_index": meta["chunk_index"],
                "chunk_text": chunk_obj.get("chunk_text", ""),
                "source_page_start": meta.get("source_page_start"),
                "source_page_end": meta.get("source_page_end"),
                "source_url": meta.get("source_url", ""),
                "source_authority": meta.get("source_authority", "India Code")
            })

        return results

    def search_reranked(self, query: str, top_k: int = 5, candidate_k: int = DEFAULT_CANDIDATE_K) -> List[Dict[str, Any]]:
        """
        Retrieves candidate_k candidates from FAISS and reranks them using transparent multi-signal scoring.
        """
        model = self._get_model()
        query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

        scores, indices = self._index.search(query_emb, candidate_k)

        query_words, query_phrase = self._extract_keywords_and_phrases(query)

        candidates = []
        for vec_idx, faiss_score in zip(indices[0], scores[0]):
            if vec_idx < 0 or vec_idx >= len(self._metadata):
                continue

            meta = self._metadata[vec_idx]
            cid = meta["chunk_id"]
            chunk_obj = self._chunks_by_id.get(cid, {})
            chunk_text = chunk_obj.get("chunk_text", "")
            section_title = meta["section_title"]

            kw_score, p_match = self.compute_reranking_signals(
                query_words=query_words,
                query_phrase=query_phrase,
                section_title=section_title,
                chunk_text=chunk_text
            )

            phrase_boost = 1.0 if p_match else 0.0

            final_score = (
                self.weight_faiss * float(faiss_score) +
                self.weight_keyword * kw_score +
                self.weight_phrase * phrase_boost
            )

            candidates.append({
                "chunk_id": cid,
                "section_number": meta["section_number"],
                "section_title": section_title,
                "chunk_index": meta["chunk_index"],
                "faiss_score": round(float(faiss_score), 4),
                "keyword_score": round(float(kw_score), 4),
                "phrase_match": p_match,
                "final_score": round(float(final_score), 4),
                "chunk_text": chunk_text,
                "source_page_start": meta.get("source_page_start"),
                "source_page_end": meta.get("source_page_end"),
                "source_url": meta.get("source_url", ""),
                "source_authority": meta.get("source_authority", "India Code")
            })

        # Deterministic sorting: sort by final_score desc, then faiss_score desc, then chunk_id asc
        candidates.sort(key=lambda x: (x["final_score"], x["faiss_score"], x["chunk_id"]), reverse=True)

        # Assign ranks to top_k reranked results
        final_results = []
        for rank_idx, c in enumerate(candidates[:top_k], start=1):
            c_copy = dict(c)
            c_copy["rank"] = rank_idx
            # Order key placement for clean JSON output matching exact schema
            ordered_item = {
                "rank": rank_idx,
                "chunk_id": c_copy["chunk_id"],
                "section_number": c_copy["section_number"],
                "section_title": c_copy["section_title"],
                "faiss_score": c_copy["faiss_score"],
                "keyword_score": c_copy["keyword_score"],
                "phrase_match": c_copy["phrase_match"],
                "final_score": c_copy["final_score"],
                "chunk_text": c_copy["chunk_text"],
                "source_page_start": c_copy["source_page_start"],
                "source_page_end": c_copy["source_page_end"],
                "source_url": c_copy["source_url"],
                "source_authority": c_copy["source_authority"]
            }
            final_results.append(ordered_item)

        return final_results


def run_stage5_pipeline(
    sections_file: Path,
    chunks_file: Path,
    embeddings_file: Path,
    metadata_file: Path,
    faiss_index_file: Path,
    output_report_file: Path,
    benchmark_queries: List[str]
) -> Dict[str, Any]:
    """
    Validates pipeline immutability, scores, deterministic behavior, and runs benchmark queries.
    """
    # 1. SHA256 Immutability Check
    hashes_before = {
        "sections": hashlib.sha256(open(sections_file, "rb").read()).hexdigest(),
        "chunks": hashlib.sha256(open(chunks_file, "rb").read()).hexdigest(),
        "embeddings": hashlib.sha256(open(embeddings_file, "rb").read()).hexdigest(),
        "metadata": hashlib.sha256(open(metadata_file, "rb").read()).hexdigest(),
        "faiss_index": hashlib.sha256(open(faiss_index_file, "rb").read()).hexdigest()
    }

    retriever = BNSRerankedRetriever(
        index_path=faiss_index_file,
        metadata_path=metadata_file,
        chunks_path=chunks_file
    )

    with open(chunks_file, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
        valid_chunk_ids = set(c["chunk_id"] for c in all_chunks)

    with open(sections_file, "r", encoding="utf-8") as f:
        all_sections = json.load(f)
        valid_section_numbers = set(str(s["section_number"]) for s in all_sections["sections"])

    all_chunk_ids_exist = True
    all_section_numbers_exist = True
    scores_numeric_and_finite = True
    deterministic_passed = True
    required_metadata_present = True
    queries_executed_successfully = True

    required_keys = [
        "rank", "chunk_id", "section_number", "section_title",
        "faiss_score", "keyword_score", "phrase_match", "final_score",
        "chunk_text", "source_page_start", "source_page_end",
        "source_url", "source_authority"
    ]

    for q in benchmark_queries:
        try:
            res1 = retriever.search_reranked(q, top_k=5)
            res2 = retriever.search_reranked(q, top_k=5)

            if len(res1) != 5 or len(res2) != 5:
                queries_executed_successfully = False

            # Check determinism (res1 == res2)
            if [r["chunk_id"] for r in res1] != [r["chunk_id"] for r in res2]:
                deterministic_passed = False

            for r in res1:
                if r["chunk_id"] not in valid_chunk_ids:
                    all_chunk_ids_exist = False
                if str(r["section_number"]) not in valid_section_numbers:
                    all_section_numbers_exist = False

                for val in [r["faiss_score"], r["keyword_score"], r["final_score"]]:
                    if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                        scores_numeric_and_finite = False

                for k in required_keys:
                    if k not in r or r[k] is None:
                        required_metadata_present = False
        except Exception as e:
            print(f"Error executing benchmark query '{q}': {e}")
            queries_executed_successfully = False

    hashes_after = {
        "sections": hashlib.sha256(open(sections_file, "rb").read()).hexdigest(),
        "chunks": hashlib.sha256(open(chunks_file, "rb").read()).hexdigest(),
        "embeddings": hashlib.sha256(open(embeddings_file, "rb").read()).hexdigest(),
        "metadata": hashlib.sha256(open(metadata_file, "rb").read()).hexdigest(),
        "faiss_index": hashlib.sha256(open(faiss_index_file, "rb").read()).hexdigest()
    }

    files_unmodified = (hashes_before == hashes_after)

    validation_passed = (
        files_unmodified and
        all_chunk_ids_exist and
        all_section_numbers_exist and
        scores_numeric_and_finite and
        deterministic_passed and
        required_metadata_present and
        queries_executed_successfully
    )

    report = {
        "faiss_index_unchanged": files_unmodified,
        "source_files_unchanged": files_unmodified,
        "all_chunk_ids_exist": all_chunk_ids_exist,
        "all_section_numbers_exist": all_section_numbers_exist,
        "scores_numeric_and_finite": scores_numeric_and_finite,
        "ranking_deterministic": deterministic_passed,
        "required_metadata_present": required_metadata_present,
        "test_queries_executed": queries_executed_successfully,
        "validation_passed": validation_passed
    }

    output_report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    sections_path = base_dir / "data" / "structured" / "BNS_sections.json"
    chunks_path = base_dir / "data" / "chunks" / "BNS_chunks.json"
    embeddings_path = base_dir / "data" / "embeddings" / "BNS_embeddings.npy"
    metadata_path = base_dir / "data" / "embeddings" / "BNS_embedding_metadata.json"
    faiss_index_path = base_dir / "data" / "embeddings" / "BNS_faiss.index"
    output_report_path = base_dir / "data" / "embeddings" / "BNS_reranking_validation_report.json"

    benchmark_queries = [
        "What is the punishment for cheating?",
        "What is murder under the Bharatiya Nyaya Sanhita?",
        "What does the law say about theft?",
        "What is criminal breach of trust?",
        "What is the punishment for murder?",
        "What is snatching?",
        "What is a dishonest intention?"
    ]

    parser = argparse.ArgumentParser(description="SalahSetu Reranking Pipeline")
    parser.add_argument("--query", type=str, help="Natural language query to test")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to return")
    args = parser.parse_args()

    print("==================================================")
    print("  SalahSetu - Stage 5 Retrieval Quality & Reranking ")
    print("==================================================")

    report = run_stage5_pipeline(
        sections_file=sections_path,
        chunks_file=chunks_path,
        embeddings_file=embeddings_path,
        metadata_file=metadata_path,
        faiss_index_file=faiss_index_path,
        output_report_file=output_report_path,
        benchmark_queries=benchmark_queries
    )

    print("\n--- RERANKING VALIDATION REPORT SUMMARY ---")
    print(f"FAISS Index Unchanged:     {report['faiss_index_unchanged']}")
    print(f"Source Files Unchanged:    {report['source_files_unchanged']}")
    print(f"All Chunk IDs Exist:       {report['all_chunk_ids_exist']}")
    print(f"All Section Numbers Exist: {report['all_section_numbers_exist']}")
    print(f"Scores Numeric & Finite:   {report['scores_numeric_and_finite']}")
    print(f"Ranking Deterministic:     {report['ranking_deterministic']}")
    print(f"Required Metadata Present: {report['required_metadata_present']}")
    print(f"Test Queries Executed:     {report['test_queries_executed']}")
    print(f"Validation Passed:         {report['validation_passed']}")
    print("==================================================")

    retriever = BNSRerankedRetriever(
        index_path=faiss_index_path,
        metadata_path=metadata_path,
        chunks_path=chunks_path
    )

    if args.query:
        print(f"\nSINGLE QUERY: \"{args.query}\"")
        before_res = retriever.search_faiss_only(args.query, top_k=args.top_k)
        after_res = retriever.search_reranked(args.query, top_k=args.top_k)

        print("\nBEFORE (FAISS Only):")
        for r in before_res:
            print(f"Rank {r['rank']} | Score: {r['score']:.4f} | Sec {r['section_number']}: {r['section_title']} ({r['chunk_id']})")

        print("\nAFTER (Reranked):")
        for r in after_res:
            print(f"Rank {r['rank']} | Final: {r['final_score']:.4f} | FAISS: {r['faiss_score']:.4f} | KW: {r['keyword_score']:.4f} | PM: {r['phrase_match']} | Sec {r['section_number']}: {r['section_title']} ({r['chunk_id']})")
    else:
        print("\n==================================================")
        print("    BEFORE vs AFTER COMPARISON FOR 7 BENCHMARKS   ")
        print("==================================================")

        for q_idx, q_text in enumerate(benchmark_queries, start=1):
            before_res = retriever.search_faiss_only(q_text, top_k=5)
            after_res = retriever.search_reranked(q_text, top_k=5)

            print(f"\nQUERY {q_idx}: \"{q_text}\"")
            print("-----------------------------------------------------------------")
            print("BEFORE (FAISS Initial Top 5):")
            for r in before_res:
                print(f"  Rank {r['rank']} | FAISS Score: {r['score']:.4f} | Sec {r['section_number']}: {r['section_title']} ({r['chunk_id']})")

            print("\nAFTER (Refined Reranked Top 5):")
            for r in after_res:
                print(f"  Rank {r['rank']} | Final: {r['final_score']:.4f} [FAISS: {r['faiss_score']:.4f} | KW: {r['keyword_score']:.4f} | PM: {r['phrase_match']}] | Sec {r['section_number']}: {r['section_title']} ({r['chunk_id']})")

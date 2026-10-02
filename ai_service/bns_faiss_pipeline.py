"""
SalahSetu - BNS FAISS Index and Semantic Retrieval Pipeline (Stage 4)

This module builds an exact-search FAISS vector index (IndexFlatIP) from normalized
BNS chunk embeddings and provides a local semantic retrieval engine mapped to chunk text and metadata.

Output Artifacts:
- data/embeddings/BNS_faiss.index
- data/embeddings/BNS_faiss_validation_report.json
"""

import json
import hashlib
import sys
import argparse
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384
EXPECTED_VECTOR_COUNT = 387


class BNSSemanticRetriever:
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        chunks_path: Path,
        model_name: str = DEFAULT_MODEL_NAME
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.chunks_path = chunks_path
        self.model_name = model_name

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

        # Load FAISS index
        self._index = faiss.read_index(str(self.index_path))

        # Load metadata list
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        # Load chunks for exact text retrieval
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            self._chunks_by_id = {c["chunk_id"]: c["chunk_text"] for c in chunks}

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Encodes query, normalizes embedding, searches FAISS index, and maps back to text & metadata.
        """
        model = self._get_model()

        # Generate query embedding with same model and normalize
        query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        query_emb = query_emb.astype(np.float32)

        # FAISS search
        scores, indices = self._index.search(query_emb, top_k)

        results = []
        for rank_idx, (score, vec_idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if vec_idx < 0 or vec_idx >= len(self._metadata):
                continue

            meta = self._metadata[vec_idx]
            chunk_id = meta["chunk_id"]
            chunk_text = self._chunks_by_id.get(chunk_id, "")

            res_item = {
                "rank": rank_idx,
                "score": float(score),
                "chunk_id": chunk_id,
                "section_number": meta["section_number"],
                "section_title": meta["section_title"],
                "chunk_index": meta["chunk_index"],
                "chunk_text": chunk_text,
                "source_page_start": meta.get("source_page_start"),
                "source_page_end": meta.get("source_page_end"),
                "source_url": meta.get("source_url", ""),
                "source_authority": meta.get("source_authority", "India Code")
            }
            results.append(res_item)

        return results


def build_faiss_index(
    embeddings_file: Path,
    output_index_file: Path
) -> faiss.Index:
    """
    Builds and saves an exact-search FAISS inner product index (IndexFlatIP)
    from L2-normalized embeddings.
    """
    embeddings = np.load(embeddings_file).astype(np.float32)
    dim = embeddings.shape[1]

    # Use IndexFlatIP because embeddings are normalized (IP == Cosine Similarity)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    output_index_file.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_index_file))

    return index


def run_stage4_pipeline(
    sections_file: Path,
    chunks_file: Path,
    embeddings_file: Path,
    metadata_file: Path,
    output_index_file: Path,
    output_report_file: Path,
    model_name: str = DEFAULT_MODEL_NAME
) -> Dict[str, Any]:
    """
    Executes FAISS index creation, validation checks, and benchmark test queries.
    """
    # Verify input files immutability before processing
    hashes_before = {
        "sections": hashlib.sha256(open(sections_file, "rb").read()).hexdigest(),
        "chunks": hashlib.sha256(open(chunks_file, "rb").read()).hexdigest(),
        "embeddings": hashlib.sha256(open(embeddings_file, "rb").read()).hexdigest(),
        "metadata": hashlib.sha256(open(metadata_file, "rb").read()).hexdigest()
    }

    # 1. Build FAISS index
    build_faiss_index(embeddings_file, output_index_file)

    # 2. Verify input files immutability after building
    hashes_after = {
        "sections": hashlib.sha256(open(sections_file, "rb").read()).hexdigest(),
        "chunks": hashlib.sha256(open(chunks_file, "rb").read()).hexdigest(),
        "embeddings": hashlib.sha256(open(embeddings_file, "rb").read()).hexdigest(),
        "metadata": hashlib.sha256(open(metadata_file, "rb").read()).hexdigest()
    }

    inputs_unmodified = (hashes_before == hashes_after)

    # 3. Load built index & metadata for validation
    faiss_index_loadable = False
    total_vectors_indexed = 0
    index_dimension = 0

    try:
        loaded_index = faiss.read_index(str(output_index_file))
        faiss_index_loadable = True
        total_vectors_indexed = int(loaded_index.ntotal)
        index_dimension = int(loaded_index.d)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_list = json.load(f)

    metadata_mapping_valid = (len(metadata_list) == total_vectors_indexed)

    # Check ordering correspondence and duplicate chunk IDs
    seen_chunk_ids = set()
    duplicate_chunk_ids = []
    ordering_corresponds = True

    if len(metadata_list) != len(chunks_list):
        ordering_corresponds = False
    else:
        for idx in range(len(metadata_list)):
            meta_id = metadata_list[idx]["chunk_id"]
            chunk_id = chunks_list[idx]["chunk_id"]

            if meta_id != chunk_id:
                ordering_corresponds = False

            if meta_id in seen_chunk_ids:
                duplicate_chunk_ids.append(meta_id)
            else:
                seen_chunk_ids.add(meta_id)

    # 4. Test query execution using retriever
    retriever = BNSSemanticRetriever(
        index_path=output_index_file,
        metadata_path=metadata_file,
        chunks_path=chunks_file,
        model_name=model_name
    )

    test_query = "What is the punishment for cheating?"
    test_results = retriever.search(test_query, top_k=5)

    test_query_passed = (
        len(test_results) == 5 and
        all(res["chunk_id"] in seen_chunk_ids for res in test_results)
    )

    validation_passed = (
        faiss_index_loadable and
        total_vectors_indexed == EXPECTED_VECTOR_COUNT and
        index_dimension == EXPECTED_DIMENSION and
        metadata_mapping_valid and
        ordering_corresponds and
        len(duplicate_chunk_ids) == 0 and
        test_query_passed and
        inputs_unmodified
    )

    report = {
        "model_name": model_name,
        "total_vectors_indexed": total_vectors_indexed,
        "index_dimension": index_dimension,
        "faiss_index_loadable": faiss_index_loadable,
        "metadata_mapping_valid": metadata_mapping_valid,
        "ordering_corresponds": ordering_corresponds,
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "query_dimension": EXPECTED_DIMENSION,
        "test_query_passed": test_query_passed,
        "validation_passed": validation_passed
    }

    # Save validation report
    with open(output_report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    sections_path = base_dir / "data" / "structured" / "BNS_sections.json"
    chunks_path = base_dir / "data" / "chunks" / "BNS_chunks.json"
    embeddings_path = base_dir / "data" / "embeddings" / "BNS_embeddings.npy"
    metadata_path = base_dir / "data" / "embeddings" / "BNS_embedding_metadata.json"
    output_index_path = base_dir / "data" / "embeddings" / "BNS_faiss.index"
    output_report_path = base_dir / "data" / "embeddings" / "BNS_faiss_validation_report.json"

    parser = argparse.ArgumentParser(description="SalahSetu FAISS Index & Retrieval Pipeline")
    parser.add_argument("--query", type=str, help="Natural language query to test")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to return")
    args = parser.parse_args()

    print("==================================================")
    print("  SalahSetu - BNS FAISS Vector Index & Retrieval  ")
    print("==================================================")

    report = run_stage4_pipeline(
        sections_file=sections_path,
        chunks_file=chunks_path,
        embeddings_file=embeddings_path,
        metadata_file=metadata_path,
        output_index_file=output_index_path,
        output_report_file=output_report_path,
        model_name=DEFAULT_MODEL_NAME
    )

    print("\n--- FAISS INDEX VALIDATION REPORT SUMMARY ---")
    print(f"Embedding Model:         {report['model_name']}")
    print(f"Total Vectors Indexed:   {report['total_vectors_indexed']}")
    print(f"Index Dimension:         {report['index_dimension']}")
    print(f"FAISS Index Loadable:    {report['faiss_index_loadable']}")
    print(f"Metadata Mapping Valid:  {report['metadata_mapping_valid']}")
    print(f"Ordering Corresponds:    {report['ordering_corresponds']}")
    print(f"Duplicate Chunk IDs:     {len(report['duplicate_chunk_ids'])}")
    print(f"Query Dimension:         {report['query_dimension']}")
    print(f"Test Query Passed:       {report['test_query_passed']}")
    print(f"Validation Passed:       {report['validation_passed']}")
    print("==================================================")

    retriever = BNSSemanticRetriever(
        index_path=output_index_path,
        metadata_path=metadata_path,
        chunks_path=chunks_path,
        model_name=DEFAULT_MODEL_NAME
    )

    if args.query:
        print(f"\nQUERY: \"{args.query}\"")
        results = retriever.search(args.query, top_k=args.top_k)
        for r in results:
            print(f"Rank {r['rank']} | Score: {r['score']:.4f} | Sec {r['section_number']}: {r['section_title']} (Chunk: {r['chunk_id']})")
    else:
        # Benchmark 4 required queries
        benchmark_queries = [
            "What is the punishment for cheating?",
            "What is murder under the Bharatiya Nyaya Sanhita?",
            "What does the law say about theft?",
            "What is criminal breach of trust?"
        ]

        print("\n==================================================")
        print("          BENCHMARK SEMANTIC SEARCH RESULTS       ")
        print("==================================================")

        for q_idx, q_text in enumerate(benchmark_queries, start=1):
            print(f"\nQUERY {q_idx}: \"{q_text}\"")
            print("-" * 65)
            results = retriever.search(q_text, top_k=5)
            for r in results:
                print(f"Rank {r['rank']} | Score: {r['score']:.4f} | Sec {r['section_number']}: {r['section_title']} (Chunk: {r['chunk_id']})")

"""
SalahSetu - BNS Legal Embeddings Pipeline (Stage 3)

This script reads structured section chunks from data/chunks/BNS_chunks.json
and converts every chunk into a normalized semantic vector embedding using a configurable
Sentence Transformers model.

Output Artifacts:
- data/embeddings/BNS_embeddings.npy
- data/embeddings/BNS_embedding_metadata.json
- data/embeddings/BNS_embedding_validation_report.json
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Configurable Embedding Parameters
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REQUIRED_METADATA_KEYS = [
    "chunk_id",
    "section_number",
    "section_title",
    "chunk_index",
    "source_page_start",
    "source_page_end",
    "source_url",
    "source_authority"
]


def load_sentence_transformer(model_name: str):
    """
    Loads sentence-transformers SentenceTransformer model.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load sentence-transformers model '{model_name}': {e}")


def generate_embeddings(
    chunks: List[Dict[str, Any]],
    model_name: str = DEFAULT_MODEL_NAME,
    normalize: bool = True
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Generates semantic vector embeddings and separate metadata for each chunk.
    
    Embeddings are normalized to unit L2 norm so dot product equals cosine similarity.
    """
    model = load_sentence_transformer(model_name)
    
    # Extract chunk texts and metadata in strict chunk order
    chunk_texts = [chunk["chunk_text"] for chunk in chunks]
    
    metadata_list = []
    for chunk in chunks:
        meta = {
            "chunk_id": chunk["chunk_id"],
            "section_number": chunk["section_number"],
            "section_title": chunk["section_title"],
            "chunk_index": chunk["chunk_index"],
            "source_page_start": chunk.get("source_page_start"),
            "source_page_end": chunk.get("source_page_end"),
            "source_url": chunk.get("source_url", ""),
            "source_authority": chunk.get("source_authority", "")
        }
        metadata_list.append(meta)

    # Encode all chunk texts into dense vectors
    embeddings = model.encode(
        chunk_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize
    )
    
    return embeddings.astype(np.float32), metadata_list


def process_bns_embeddings(
    chunks_file: Path,
    output_embeddings_file: Path,
    output_metadata_file: Path,
    output_report_file: Path,
    sections_file: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    normalize: bool = True
) -> Dict[str, Any]:
    """
    Executes the legal embedding pipeline and validation checks.
    """
    # Verify source-of-truth files before processing
    with open(sections_file, "rb") as f:
        sections_hash_before = hashlib.sha256(f.read()).hexdigest()
    with open(chunks_file, "rb") as f:
        chunks_hash_before = hashlib.sha256(f.read()).hexdigest()

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    total_chunks = len(chunks)

    # Generate embeddings and metadata
    embeddings, metadata_records = generate_embeddings(
        chunks=chunks,
        model_name=model_name,
        normalize=normalize
    )

    # Verify source-of-truth files after processing
    with open(sections_file, "rb") as f:
        sections_hash_after = hashlib.sha256(f.read()).hexdigest()
    with open(chunks_file, "rb") as f:
        chunks_hash_after = hashlib.sha256(f.read()).hexdigest()

    sources_unmodified = (
        sections_hash_before == sections_hash_after and
        chunks_hash_before == chunks_hash_after
    )

    # Perform Validation Checks
    total_embeddings = len(embeddings)
    embedding_dimension = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0

    invalid_vectors_count = int(
        np.isnan(embeddings).sum() + np.isinf(embeddings).sum()
    )

    seen_ids = set()
    duplicate_chunk_ids = []
    missing_metadata = []

    ordering_valid = True
    if len(metadata_records) != total_chunks:
        ordering_valid = False
    else:
        for idx in range(total_chunks):
            chunk_id = chunks[idx]["chunk_id"]
            meta_id = metadata_records[idx]["chunk_id"]

            if chunk_id != meta_id:
                ordering_valid = False

            if meta_id in seen_ids:
                duplicate_chunk_ids.append(meta_id)
            else:
                seen_ids.add(meta_id)

            for key in REQUIRED_METADATA_KEYS:
                if key not in metadata_records[idx] or metadata_records[idx][key] is None:
                    missing_metadata.append(f"{meta_id}:{key}")

    validation_passed = (
        total_chunks == 387 and
        total_embeddings == 387 and
        len(metadata_records) == 387 and
        embedding_dimension > 0 and
        len(duplicate_chunk_ids) == 0 and
        len(missing_metadata) == 0 and
        invalid_vectors_count == 0 and
        ordering_valid and
        sources_unmodified
    )

    report = {
        "model_name": model_name,
        "total_chunks": total_chunks,
        "total_embeddings": total_embeddings,
        "embedding_dimension": embedding_dimension,
        "metadata_records": len(metadata_records),
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "missing_metadata": missing_metadata,
        "invalid_vectors": invalid_vectors_count,
        "ordering_valid": ordering_valid,
        "validation_passed": validation_passed
    }

    # Save output artifacts
    output_embeddings_file.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_embeddings_file, embeddings)

    with open(output_metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2, ensure_ascii=False)

    with open(output_report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    sections_path = base_dir / "data" / "structured" / "BNS_sections.json"
    chunks_path = base_dir / "data" / "chunks" / "BNS_chunks.json"
    output_embeddings_path = base_dir / "data" / "embeddings" / "BNS_embeddings.npy"
    output_metadata_path = base_dir / "data" / "embeddings" / "BNS_embedding_metadata.json"
    output_report_path = base_dir / "data" / "embeddings" / "BNS_embedding_validation_report.json"

    print("==================================================")
    print("     SalahSetu - BNS Legal Embeddings Pipeline    ")
    print("==================================================")
    print(f"Chunks Input:      {chunks_path}")
    print(f"Embeddings Output: {output_embeddings_path}")
    print(f"Metadata Output:   {output_metadata_path}")
    print(f"Report Output:     {output_report_path}")
    print(f"Embedding Model:   {DEFAULT_MODEL_NAME}")
    print("--------------------------------------------------")

    report = process_bns_embeddings(
        chunks_file=chunks_path,
        output_embeddings_file=output_embeddings_path,
        output_metadata_file=output_metadata_path,
        output_report_file=output_report_path,
        sections_file=sections_path,
        model_name=DEFAULT_MODEL_NAME,
        normalize=True
    )

    print("\n--- EMBEDDING VALIDATION REPORT SUMMARY ---")
    print(f"Selected Model:      {report['model_name']}")
    print(f"Total Chunks:        {report['total_chunks']}")
    print(f"Total Embeddings:    {report['total_embeddings']}")
    print(f"Embedding Dimension: {report['embedding_dimension']}")
    print(f"Metadata Records:    {report['metadata_records']}")
    print(f"Duplicate Chunk IDs: {len(report['duplicate_chunk_ids'])}")
    print(f"Missing Metadata:    {len(report['missing_metadata'])}")
    print(f"Invalid Vectors:     {report['invalid_vectors']}")
    print(f"Ordering Valid:      {report['ordering_valid']}")
    print(f"Validation Passed:   {report['validation_passed']}")
    print("==================================================")

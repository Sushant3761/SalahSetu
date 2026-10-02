"""
SalahSetu - BNS Legal Chunking Pipeline (Stage 2)

This script reads structured section data from data/structured/BNS_sections.json
and generates deterministic, context-aware legal chunks saved to data/chunks/BNS_chunks.json
along with a validation report saved to data/chunks/BNS_chunk_validation_report.json.
"""

import json
import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any

# Configurable Chunking Parameters
DEFAULT_MAX_CHUNK_SIZE = 3000
DEFAULT_OVERLAP_SIZE = 300

REQUIRED_CHUNK_METADATA_KEYS = [
    "chunk_id",
    "act_name",
    "act_number",
    "section_number",
    "section_title",
    "chunk_index",
    "chunk_text",
    "source_page_start",
    "source_page_end",
    "source_url",
    "source_authority"
]


def split_text_into_legal_blocks(text: str) -> List[str]:
    """
    Splits section text at legal structural boundaries (subsections, clauses, explanations, exceptions, illustrations)
    while preserving original wording and newline markers.
    Joining all returned blocks reconstructs the exact original text.
    """
    pattern = r'\n(?=\(\d+\)|\([a-z]\)|\([ivx]+\)|Explanation|Exception|Illustration|Provided|\n)'
    split_indices = [0] + [m.start() for m in re.finditer(pattern, text)] + [len(text)]
    blocks = [text[split_indices[i]:split_indices[i+1]] for i in range(len(split_indices) - 1)]
    
    # If any individual block happens to exceed MAX_CHUNK_SIZE, fall back to line splitting for that block
    final_blocks = []
    for block in blocks:
        if len(block) > DEFAULT_MAX_CHUNK_SIZE:
            line_indices = [0] + [m.start() for m in re.finditer(r'\n', block)] + [len(block)]
            sub_blocks = [block[line_indices[i]:line_indices[i+1]] for i in range(len(line_indices) - 1)]
            final_blocks.extend(sub_blocks)
        else:
            final_blocks.append(block)
            
    return final_blocks


def chunk_section(
    section: Dict[str, Any],
    root_metadata: Dict[str, Any],
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    overlap_size: int = DEFAULT_OVERLAP_SIZE
) -> List[Dict[str, Any]]:
    """
    Generates deterministic chunks for a single legal section.
    """
    sec_num = str(section["section_number"])
    sec_title = section.get("section_title", "")
    sec_text = section.get("section_text", "")
    
    act_name = section.get("act_name") or root_metadata.get("act_name", "")
    act_number = section.get("act_number") or root_metadata.get("act_number", "")
    source_url = section.get("source_url") or root_metadata.get("source_url", "")
    source_authority = section.get("source_authority") or root_metadata.get("source_authority", "")
    source_page_start = section.get("source_page_start")
    source_page_end = section.get("source_page_end")

    # If section fits within max_chunk_size, return single chunk
    if len(sec_text) <= max_chunk_size:
        chunk = {
            "chunk_id": f"BNS_SEC_{sec_num}_CHUNK_0",
            "act_name": act_name,
            "act_number": act_number,
            "section_number": sec_num,
            "section_title": sec_title,
            "chunk_index": 0,
            "chunk_text": sec_text,
            "source_page_start": source_page_start,
            "source_page_end": source_page_end,
            "source_url": source_url,
            "source_authority": source_authority
        }
        return [chunk]

    # For long sections, split into structural blocks and assemble chunks with overlap
    blocks = split_text_into_legal_blocks(sec_text)
    chunks = []
    chunk_index = 0
    curr_start = 0

    while curr_start < len(blocks):
        curr_chunk_blocks = [blocks[curr_start]]
        curr_len = len(blocks[curr_start])
        end_idx = curr_start

        for j in range(curr_start + 1, len(blocks)):
            if curr_len + len(blocks[j]) <= max_chunk_size:
                curr_chunk_blocks.append(blocks[j])
                curr_len += len(blocks[j])
                end_idx = j
            else:
                break

        chunk_text = "".join(curr_chunk_blocks)
        chunks.append({
            "chunk_id": f"BNS_SEC_{sec_num}_CHUNK_{chunk_index}",
            "act_name": act_name,
            "act_number": act_number,
            "section_number": sec_num,
            "section_title": sec_title,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "source_page_start": source_page_start,
            "source_page_end": source_page_end,
            "source_url": source_url,
            "source_authority": source_authority
        })
        chunk_index += 1

        if end_idx == len(blocks) - 1:
            break

        # Calculate overlap for next_start
        next_start = end_idx
        ov_len = 0
        for k in range(end_idx, curr_start, -1):
            blen = len(blocks[k])
            if ov_len + blen <= overlap_size or k == end_idx:
                ov_len += blen
                next_start = k
            else:
                break
                
        if next_start <= curr_start:
            next_start = curr_start + 1
            
        curr_start = next_start

    return chunks


def process_bns_chunking(
    input_file: Path,
    output_chunks_file: Path,
    output_report_file: Path,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    overlap_size: int = DEFAULT_OVERLAP_SIZE
) -> Dict[str, Any]:
    """
    Executes the BNS legal chunking pipeline and validation checks.
    """
    # Hash input file before processing to guarantee no modification occurs
    with open(input_file, "rb") as f:
        input_hash_before = hashlib.sha256(f.read()).hexdigest()

    with open(input_file, "r", encoding="utf-8") as f:
        root_data = json.load(f)

    sections = root_data.get("sections", [])
    total_source_sections = len(sections)

    root_metadata = {
        "act_name": root_data.get("act_name", ""),
        "act_number": root_data.get("act_number", ""),
        "source_url": root_data.get("source_url", ""),
        "source_authority": root_data.get("source_authority", "")
    }

    all_chunks = []
    for sec in sections:
        sec_chunks = chunk_section(sec, root_metadata, max_chunk_size, overlap_size)
        all_chunks.extend(sec_chunks)

    # Hash input file after processing
    with open(input_file, "rb") as f:
        input_hash_after = hashlib.sha256(f.read()).hexdigest()

    file_unmodified = (input_hash_before == input_hash_after)

    # Perform Validation Checks
    expected_sections = set(str(i) for i in range(1, 359))
    found_sections = set(c["section_number"] for c in all_chunks)

    missing_sections = sorted(list(expected_sections - found_sections), key=lambda x: int(x))
    unexpected_sections = sorted(list(found_sections - expected_sections), key=lambda x: int(x) if x.isdigit() else x)

    seen_ids = set()
    duplicate_chunk_ids = []
    empty_chunks = []
    missing_metadata = []

    for c in all_chunks:
        cid = c.get("chunk_id")
        if cid in seen_ids:
            duplicate_chunk_ids.append(cid)
        else:
            seen_ids.add(cid)

        if not c.get("chunk_text") or not c["chunk_text"].strip():
            empty_chunks.append(cid)

        for k in REQUIRED_CHUNK_METADATA_KEYS:
            if k not in c or c[k] is None:
                missing_metadata.append(f"{cid}:{k}")

    validation_passed = (
        total_source_sections == 358 and
        len(missing_sections) == 0 and
        len(unexpected_sections) == 0 and
        len(duplicate_chunk_ids) == 0 and
        len(empty_chunks) == 0 and
        len(missing_metadata) == 0 and
        file_unmodified
    )

    report = {
        "total_source_sections": total_source_sections,
        "total_chunks": len(all_chunks),
        "missing_sections": missing_sections,
        "unexpected_sections": unexpected_sections,
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "empty_chunks": empty_chunks,
        "missing_metadata": missing_metadata,
        "validation_passed": validation_passed
    }

    # Ensure output directory exists
    output_chunks_file.parent.mkdir(parents=True, exist_ok=True)

    # Write generated chunks
    with open(output_chunks_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Write validation report
    with open(output_report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    input_path = base_dir / "data" / "structured" / "BNS_sections.json"
    output_chunks_path = base_dir / "data" / "chunks" / "BNS_chunks.json"
    output_report_path = base_dir / "data" / "chunks" / "BNS_chunk_validation_report.json"

    print("==================================================")
    print("      SalahSetu - BNS Legal Chunking Pipeline     ")
    print("==================================================")
    print(f"Input file:  {input_path}")
    print(f"Chunks file: {output_chunks_path}")
    print(f"Report file: {output_report_path}")
    print("--------------------------------------------------")

    report = process_bns_chunking(
        input_file=input_path,
        output_chunks_file=output_chunks_path,
        output_report_file=output_report_path,
        max_chunk_size=DEFAULT_MAX_CHUNK_SIZE,
        overlap_size=DEFAULT_OVERLAP_SIZE
    )

    print("\n--- VALIDATION REPORT SUMMARY ---")
    print(f"Total Source Sections: {report['total_source_sections']}")
    print(f"Total Chunks Generated: {report['total_chunks']}")
    print(f"Missing Sections:      {len(report['missing_sections'])}")
    print(f"Unexpected Sections:   {len(report['unexpected_sections'])}")
    print(f"Duplicate Chunk IDs:   {len(report['duplicate_chunk_ids'])}")
    print(f"Empty Chunks:          {len(report['empty_chunks'])}")
    print(f"Missing Metadata:      {len(report['missing_metadata'])}")
    print(f"Validation Passed:     {report['validation_passed']}")
    print("==================================================")

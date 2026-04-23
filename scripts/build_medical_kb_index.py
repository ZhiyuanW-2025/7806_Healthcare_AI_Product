#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def sent_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def simple_tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())


def chunk_text(text: str, max_chars: int = 620, overlap_sentences: int = 1) -> List[str]:
    sents = sent_split(text)
    if not sents:
        return []
    chunks: List[str] = []
    i = 0
    while i < len(sents):
        current: List[str] = []
        total = 0
        j = i
        while j < len(sents):
            s = sents[j]
            add = len(s) + (1 if total > 0 else 0)
            if total + add > max_chars and current:
                break
            current.append(s)
            total += add
            j += 1
        chunks.append(" ".join(current).strip())
        if j >= len(sents):
            break
        i = max(i + 1, j - overlap_sentences)
    return chunks


def build_index(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for doc in documents:
        chunks = chunk_text(str(doc.get("content", "")))
        if not chunks:
            continue
        for idx, chunk in enumerate(chunks, start=1):
            metadata_text = " ".join(
                [
                    str(doc.get("title", "")),
                    str(doc.get("topic", "")),
                    " ".join(doc.get("subtopics", []) or []),
                    " ".join(doc.get("audience", []) or []),
                    str(doc.get("otc_guidance", "")),
                    " ".join(doc.get("red_flags", []) or []),
                    " ".join(doc.get("contraindications", []) or []),
                    str(doc.get("citation_snippet", "")),
                ]
            )
            rows.append(
                {
                    "chunk_id": f"{doc.get('id', 'doc')}_c{idx}",
                    "doc_id": doc.get("id", ""),
                    "title": doc.get("title", ""),
                    "topic": doc.get("topic", ""),
                    "subtopics": doc.get("subtopics", []),
                    "issuer": doc.get("issuer", ""),
                    "jurisdiction": doc.get("jurisdiction", "EU-general"),
                    "updated_at": doc.get("updated_at", ""),
                    "language": doc.get("language", "en"),
                    "audience": doc.get("audience", []),
                    "source_url": doc.get("source_url", ""),
                    "version": doc.get("version", ""),
                    "chunk_text": chunk,
                    "otc_guidance": doc.get("otc_guidance", ""),
                    "contraindications": doc.get("contraindications", []),
                    "red_flags": doc.get("red_flags", []),
                    "citation_snippet": doc.get("citation_snippet", ""),
                    "terms": simple_tokens(metadata_text + " " + chunk),
                }
            )
    return {
        "schema_version": "medical-kb-index-v1",
        "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "chunk_count": len(rows),
        "chunks": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunk index for RuralLink medical KB")
    parser.add_argument("--input", required=True, help="Path to medical_kb.json")
    parser.add_argument("--output", required=True, help="Path to write medical_kb_index.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    docs = payload.get("documents", [])
    index = build_index(docs)
    output_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Built index: {output_path} | chunks={index['chunk_count']}")


if __name__ == "__main__":
    main()

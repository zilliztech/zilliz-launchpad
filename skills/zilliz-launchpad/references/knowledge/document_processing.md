# Document processing / chunking

## Supported input shapes

Phase 1 Collect accepts the following file shapes. Downstream chunking is unchanged — the recursive splitter still runs over whatever `text` field Collect emits.

| Suffix | `data_shape` | Records emitted | Notes |
|--------|--------------|-----------------|-------|
| `.jsonl` / `.ndjson` | `jsonl` | one per line | inferred fields |
| `.csv` | `csv` | one per row | inferred fields |
| `.txt` | `text` | one per file | whole file as `text` |
| `.md` | `markdown` | one per file (default), or one per `## ` section with `--split-markdown-headings` | a leading `---\n…\n---\n` YAML front-matter block is stripped |
| `.pdf` | `pdf` | one per page (1-indexed `page_number`, plus `source_path`) | requires `pip install zilliz-launchpad[documents]`; surfaces a warning when no extractable text is found (likely a scanned PDF — add an OCR step yourself) |
| directory of images | `image_dir` | one per image | see `image_embedding_models.md` |

`--split-markdown-headings` is opt-in because most onboarding docs are short enough that one-record-per-file is the right default. Turn it on for longer hand-written notes structured around `##` sections.

## Default strategy

Recursive character splitter targeting **~512 approximate tokens** with **~64 token overlap**. Separators (in order): paragraph (`\n\n`), line (`\n`), sentence (`. `), word (` `), character (hard cut).

This is a good starting point for most prose. Override via `plan.chunking`.

## When to change

- **Short records already** (tweets, product titles, one-line logs): disable chunking — set size to a large number so each record fits whole.
- **Very long documents** (legal, medical): 1024 / 128 is more standard for retrieval-focused reading.
- **Code**: language-aware splitters (tree-sitter based) outperform character splitters. Not built into MVP — chunk code as text at size=1024/overlap=128 as a decent fallback.
- **Tables / CSV rows**: don't chunk; embed each row as a semi-structured string.

## Metadata attached to every chunk

Every chunk carries:

- `id`: deterministic SHA-256 hash of `source_id + "::" + chunk_index`, first 32 hex
- `text`: the chunk contents (post-overlap)
- Any extra scalar fields listed in the plan

## Overlap — why?

Overlap prevents a retrieval-relevant sentence from being split across chunks with no redundancy. 10-15% is typical; too much overlap wastes storage.

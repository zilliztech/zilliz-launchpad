# Document processing / chunking

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

---
title: Sample document
author: zilliz-launchpad
tags: [test, fixture]
---

# Sample document

A short prose preamble before the first `##` heading. The heading-split
reader discards this preamble; the whole-file reader keeps it.

## Introduction

Vector search starts with a sample document. This fixture exercises the
Phase 1 Collect Markdown reader, including front-matter stripping and
the optional `## ` heading split.

## Approach

The default Markdown reader treats the entire file as one record so a
README-style document round-trips with no surprises. The
`--split-markdown-headings` flag emits one record per top-level heading,
which is useful for longer hand-written notes.

## Notes

Chunking still happens later, in Phase 4 — this reader only decides the
record boundary, not the embedding-token boundary.

# Manual smoke — image upload in the demo UI

The Next.js UI is excluded from automated pytest/mypy coverage and the drag-and-drop behaviour only exercises the full stack in a real browser. This checklist is the humans-only path.

## Preconditions

- An image-search run directory with a populated collection. Quickest way:
  ```bash
  uv run python -m zilliz_launchpad.cli collect --input ./tests/fixtures/photos/ --use-case image-search --dataset-size 20 --deployment local-standalone
  # …configure / plan / execute — see README "Image search" walkthrough
  ```
- CLIP weights pre-fetched (`uv run python -c "from lib.embeddings import prefetch_clip; prefetch_clip()"`).
- An extra query image on disk — ideally something visually similar to but distinct from the ingested set (e.g. a different dog photo if the corpus is dogs).

## Steps

1. **Start the sidecar** pointed at the run dir:
   ```bash
   LAUNCHPAD_RUN_DIR=./runs/<id> uv run uvicorn lib.ui:app --port 8000
   ```
   Expect `/health` to return `{"status":"ok", ...}`. Expect `/info` to return `modality: "image"` and a non-null `embedding.model`.

2. **Start the Next.js dev server**:
   ```bash
   cd skills/zilliz-launchpad/scripts/ui && pnpm dev
   ```
   Open `http://localhost:3000` in a browser.

3. **Verify the upload control appears.** The heading should read `Image search — <collection>` and a `Search by image…` button should be visible next to `Search`. A tip line reads `Tip: drop an image anywhere on this page…`.

4. **File-picker path.** Click `Search by image…`, pick the query image. Expect:
   - The spinner shows `…` on the Search button briefly.
   - The result grid updates with ranked thumbnails.
   - A `Last image query: <filename>` line appears under the form.
   - No error message is shown.

5. **Drag-and-drop path.** Drag the same image from Finder/Explorer onto the page. Expect:
   - While dragging, the tip text switches to `Drop image to search` in a lighter blue colour.
   - On drop, results update identically to the picker path.

6. **Non-image rejection.** Drag a `.txt` or `.pdf` onto the page. Expect:
   - An inline red error: `Unsupported file type: …`.
   - The grid stays unchanged (still shows the previous image query results).
   - No network call is made (check the Network tab — `/search_image` should NOT be hit).

7. **Oversize upload.** If easy to produce, pick a >10 MB image. Expect:
   - Red inline error with the 10 MB cap message from the sidecar.
   - The previous grid remains visible.

8. **Text search still works.** With the image run active, type a text query (e.g., "sunset") into the text input and click `Search`. The existing text-to-image path should work unchanged.

## Rollback / cleanup

- Stop dev server: Ctrl-C in the pnpm window.
- Stop sidecar: Ctrl-C in the uvicorn window.
- Collection and run dir persist — re-use them for the CLI `evaluate --query-image` smoke in §4.

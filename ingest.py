"""
ingest.py  -  Multimedia RAG ingestion for DGX-1 whitepaper
FIXED v4:
  - Spatial y-coordinate caption<->image linking
  - Skip logo pages (cover=1, copyright=43)
  - Each caption claimed by at most ONE image (no duplicates)
  - Vector figure fix: render the PAGE AFTER the caption page (where diagram actually lives)
  - Smarter vector page selection: scan forward up to 3 pages for a page
    that has drawing commands but no/few text blocks (= diagram page)
"""

import fitz
import redis
import uuid
import re
import base64
import numpy as np
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
REDIS_HOST  = "redis"
REDIS_PORT  = 6379
INDEX_NAME  = "idx:docs"
EMBED_MODEL = "intfloat/e5-large-v2"
VECTOR_DIM  = 1024
CHUNK_SIZE  = 5
CHUNK_STEP  = 3

CAPTION_SEARCH_BELOW_PX = 120
MIN_IMAGE_BYTES         = 8_000

# Pages (1-based) that only have logos — skip entirely
SKIP_PAGES = {1, 43}

# DPI for rendering vector pages as PNG
RENDER_DPI = 150

r     = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
model = SentenceTransformer(EMBED_MODEL)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def embed(text: str) -> bytes:
    vec = model.encode(f"passage: {text}", normalize_embeddings=True)
    return np.array(vec, dtype=np.float32).tobytes()

def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode()

_CAPTION_RE = re.compile(
    r"(Figure|FIGURE|Fig\.?)\s*(\d+)[.:>\u2013\-]?\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)

def extract_caption(text: str):
    m = _CAPTION_RE.search(text)
    if m:
        return int(m.group(2)), m.group(3).strip()[:500]
    return None, None


def create_index():
    try:
        r.execute_command("FT.DROPINDEX", INDEX_NAME, "DD")
        print("Dropped old index")
    except Exception:
        pass
    r.execute_command(
        "FT.CREATE", INDEX_NAME,
        "ON", "HASH",
        "PREFIX", 1, "doc:",
        "SCHEMA",
        "type",           "TAG",
        "page",           "NUMERIC", "SORTABLE",
        "figure_number",  "NUMERIC", "SORTABLE",
        "content",        "TEXT",    "WEIGHT", 1.0,
        "embedding",      "VECTOR",  "FLAT",
            6,
            "TYPE",      "FLOAT32",
            "DIM",       str(VECTOR_DIM),
            "DISTANCE_METRIC", "COSINE",
    )
    print("Redis FT index created")


# ──────────────────────────────────────────────
# SPATIAL CAPTION FINDER
# ──────────────────────────────────────────────

def find_caption_for_image(img_bbox, available_caps: list) -> dict | None:
    """
    Match image to caption using bounding box y-position.
    Only considers unclaimed captions.
    """
    if not available_caps:
        return None

    img_y1 = img_bbox.y1
    img_y0 = img_bbox.y0

    # Pass 1: caption just below image
    below_near = [
        c for c in available_caps
        if img_y1 <= c["y0"] <= img_y1 + CAPTION_SEARCH_BELOW_PX
    ]
    if below_near:
        return min(below_near, key=lambda c: c["y0"] - img_y1)

    # Pass 2: any caption below image on same page
    below_any = [c for c in available_caps if c["y0"] >= img_y1]
    if below_any:
        return min(below_any, key=lambda c: c["y0"])

    # Pass 3: caption above image
    above = [c for c in available_caps if c["y1"] <= img_y0]
    if above:
        return max(above, key=lambda c: c["y1"])

    return None


# ──────────────────────────────────────────────
# VECTOR FIGURE PAGE FINDER  ← KEY FIX
# ──────────────────────────────────────────────

def find_diagram_page(doc, caption_page_0indexed: int, max_lookahead: int = 3) -> int:
    """
    For a caption on `caption_page_0indexed`, find the actual page that contains
    the diagram. In many PDFs, the caption text is on page N but the vector
    diagram is on page N+1 (or even same page in a different region).

    Strategy:
    1. Check if caption page itself has significant drawing commands
       (paths/curves) but relatively little text → diagram is here
    2. Scan forward up to max_lookahead pages for a page with:
       - drawing commands (get_drawings returns non-empty list)  
       - fewer than 8 text blocks (mostly diagram, not text)
    3. Fall back to caption page itself
    """
    total_pages = len(doc)

    def page_has_diagram(page) -> bool:
        """True if page looks like it contains a vector diagram."""
        drawings = page.get_drawings()
        if not drawings:
            return False
        # Count meaningful text blocks (skip headers/footers < 50 chars)
        blocks = page.get_text("blocks")
        real_text_blocks = [b for b in blocks if b[6] == 0 and len(b[4].strip()) > 50]
        # A diagram page has drawings AND relatively few text blocks
        return len(drawings) > 5 and len(real_text_blocks) < 12

    # Check caption page itself first
    cap_page = doc[caption_page_0indexed]
    if page_has_diagram(cap_page):
        return caption_page_0indexed

    # Scan forward
    for offset in range(1, max_lookahead + 1):
        idx = caption_page_0indexed + offset
        if idx >= total_pages:
            break
        page = doc[idx]
        if page_has_diagram(page):
            print(f"    → vector diagram found on page {idx + 1} (caption was on {caption_page_0indexed + 1})")
            return idx

    # Fall back to caption page
    return caption_page_0indexed


def render_page_as_png(doc, page_0indexed: int) -> bytes:
    """Render a full PDF page to PNG bytes at RENDER_DPI."""
    page = doc[page_0indexed]
    mat  = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


# ──────────────────────────────────────────────
# INGESTION
# ──────────────────────────────────────────────

def ingest_pdf(pdf_path: str):
    create_index()
    doc  = fitz.open(pdf_path)
    pipe = r.pipeline(transaction=False)

    # ── Pass 1: collect text blocks and captions per page ──────────
    pages_data = {}

    for page_number in range(len(doc)):
        page = doc[page_number]
        p    = page_number + 1
        raw_blocks = page.get_text("blocks")

        text_blocks    = []
        caption_blocks = []

        for blk in raw_blocks:
            if blk[6] != 0:
                continue
            t = blk[4].strip()
            if len(t) < 10:
                continue

            fig_num, cap_title = extract_caption(t)
            if fig_num is not None:
                cap_text = f"Figure {fig_num}: {cap_title}"
                caption_blocks.append({
                    "figure_number": fig_num,
                    "title":         cap_title,
                    "content":       cap_text,
                    "y0":            blk[1],
                    "y1":            blk[3],
                    "page":          p,
                    "page_0idx":     page_number,
                    "claimed":       False,
                })
                nid = str(uuid.uuid4())
                pipe.hset(f"doc:{nid}", mapping={
                    "type":          "CAPTION",
                    "figure_number": fig_num,
                    "page":          p,
                    "content":       cap_text,
                    "embedding":     embed(cap_text),
                })
            else:
                text_blocks.append(t)

        pages_data[p] = {
            "text_blocks":    text_blocks,
            "caption_blocks": caption_blocks,
        }

    print(f"Parsed {len(doc)} pages")

    # ── Pass 2: sliding-window text chunks ─────────────────────────
    for p, data in pages_data.items():
        texts = data["text_blocks"]
        if not texts:
            continue
        if len(texts) <= CHUNK_SIZE:
            chunks = [" ".join(texts)]
        else:
            chunks = []
            for i in range(0, len(texts) - CHUNK_SIZE + 1, CHUNK_STEP):
                chunks.append(" ".join(texts[i : i + CHUNK_SIZE]))
            remainder = texts[-(len(texts) % CHUNK_STEP or CHUNK_STEP):]
            if remainder and " ".join(remainder) not in chunks:
                chunks.append(" ".join(remainder))
        for chunk in chunks:
            nid = str(uuid.uuid4())
            pipe.hset(f"doc:{nid}", mapping={
                "type":      "TEXT",
                "page":      p,
                "content":   chunk[:2000],
                "embedding": embed(chunk[:512]),
            })

    # Build flat caption list for cross-page fallback
    all_captions_flat = []
    for p, data in pages_data.items():
        all_captions_flat.extend(data["caption_blocks"])
    all_captions_flat.sort(key=lambda c: (c["page"], c["figure_number"]))

    # ── Pass 3: raster image extraction ────────────────────────────
    img_count     = 0
    matched_count = 0
    skipped_count = 0

    for page_number in range(len(doc)):
        page = doc[page_number]
        p    = page_number + 1

        if p in SKIP_PAGES:
            skipped_count += 1
            continue

        img_list  = page.get_images(full=True)
        page_caps = pages_data[p]["caption_blocks"]

        for img in img_list:
            xref       = img[0]
            base_image = doc.extract_image(xref)
            img_bytes  = base_image["image"]

            if len(img_bytes) < MIN_IMAGE_BYTES:
                skipped_count += 1
                continue

            img_count += 1
            linked_cap  = None
            match_method = "none"

            # Strategy 1: spatial match on same page (unclaimed only)
            try:
                img_bbox  = page.get_image_bbox(img)
                unclaimed = [c for c in page_caps if not c["claimed"]]
                linked_cap = find_caption_for_image(img_bbox, unclaimed)
                if linked_cap:
                    linked_cap["claimed"] = True
                    match_method = "spatial-same-page"
                    matched_count += 1
            except Exception as e:
                print(f"  bbox error page {p}: {e}")

            # Strategy 2: any unclaimed caption on same page
            if not linked_cap:
                unclaimed = [c for c in page_caps if not c["claimed"]]
                if unclaimed:
                    linked_cap = unclaimed[0]
                    linked_cap["claimed"] = True
                    match_method = "unclaimed-same-page"
                    matched_count += 1

            # Strategy 3: nearest unclaimed caption within ±2 pages
            if not linked_cap:
                nearby = [
                    c for c in all_captions_flat
                    if not c["claimed"] and abs(c["page"] - p) <= 2
                ]
                if nearby:
                    linked_cap = min(nearby, key=lambda c: (abs(c["page"] - p), c["figure_number"]))
                    linked_cap["claimed"] = True
                    match_method = "adjacent-page"
                    matched_count += 1

            if linked_cap:
                cap_text   = linked_cap["content"]
                fig_number = linked_cap["figure_number"]
                print(f"  Page {p}: img#{img_count} -> Figure {fig_number} [{match_method}] {cap_text[:60]}")
            else:
                cap_text   = f"diagram on page {p}"
                fig_number = -1
                print(f"  Page {p}: img#{img_count} -> UNMATCHED")

            nid = str(uuid.uuid4())
            pipe.hset(f"doc:{nid}", mapping={
                "type":          "IMAGE",
                "page":          p,
                "figure_number": fig_number,
                "content":       image_to_base64(img_bytes),
                "embedding":     embed(cap_text),
            })

    # ── Pass 4: vector figures — unclaimed captions ─────────────────
    # These are figures drawn as PDF vector paths (not raster images).
    # We find the actual diagram page by looking for drawing commands,
    # then render it as a PNG.
    print("\nChecking for vector-only figures (unclaimed captions)...")
    vector_count = 0

    for cap in all_captions_flat:
        if cap["claimed"]:
            continue
        if cap["page"] in SKIP_PAGES:
            continue

        fig_number   = cap["figure_number"]
        cap_page_idx = cap["page_0idx"]   # 0-indexed

        # Find the page that actually contains the diagram
        diagram_page_idx = find_diagram_page(doc, cap_page_idx, max_lookahead=3)
        diagram_page_1   = diagram_page_idx + 1  # 1-based for display

        png_bytes  = render_page_as_png(doc, diagram_page_idx)
        cap_text   = cap["content"]
        cap["claimed"] = True
        vector_count  += 1

        print(f"  Caption page {cap['page']}: Figure {fig_number} [vector-render page {diagram_page_1}] {cap_text[:55]}")

        nid = str(uuid.uuid4())
        pipe.hset(f"doc:{nid}", mapping={
            "type":          "IMAGE",
            "page":          diagram_page_1,
            "figure_number": fig_number,
            "content":       image_to_base64(png_bytes),
            "embedding":     embed(cap_text),
        })

    pipe.execute()

    total_figs = matched_count + vector_count
    print(f"\nIngestion complete")
    print(f"  Pages         : {len(doc)}")
    print(f"  Raster images : {img_count}  matched={matched_count}  unmatched={img_count - matched_count}")
    print(f"  Vector figs   : {vector_count} (page-rendered)")
    print(f"  Skipped       : {skipped_count} (logos/icons)")
    print(f"  Total figures : {total_figs}")


if __name__ == "__main__":
    ingest_pdf("dgx1-v100-system-architecture-whitepaper.pdf")
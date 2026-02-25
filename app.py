"""
app.py  -  Multimedia Hybrid RAG  (FIXED)
Key fixes:
  - /debug-figures endpoint to verify Redis contents
  - /health endpoint
  - Better logging so you can see exactly what's happening
"""

import os
import re
import json
import redis
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
REDIS_HOST      = "redis"
REDIS_PORT      = 6379
INDEX_NAME      = "idx:docs"
EMBEDDING_MODEL = "intfloat/e5-large-v2"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

r           = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
model       = SentenceTransformer(EMBEDDING_MODEL)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ─────────────────────────────────────────────
# HEALTH + DEBUG
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        r.ping()
        res   = r.execute_command("FT.SEARCH", INDEX_NAME, "*", "LIMIT", 0, 0)
        total = res[0] if res else 0
        return {"status": "ok", "redis": "connected", "total_docs": int(total)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/debug-figures")
def debug_figures():
    """
    Visit http://localhost:8000/debug-figures to verify ingest worked.
    Shows every CAPTION and IMAGE stored, with their figure_number.
    """
    try:
        cap_res = r.execute_command(
            "FT.SEARCH", INDEX_NAME, "@type:{CAPTION}",
            "LIMIT", 0, 100,
            "RETURN", 3, "content", "page", "figure_number",
            "DIALECT", 2,
        )
        captions = _parse(cap_res)

        img_res = r.execute_command(
            "FT.SEARCH", INDEX_NAME, "@type:{IMAGE}",
            "LIMIT", 0, 100,
            "RETURN", 3, "page", "figure_number", "content",
            "DIALECT", 2,
        )
        images_raw = _parse(img_res)

        images = []
        for img in images_raw:
            b64 = img.get("content", "")
            images.append({
                "page":          img.get("page"),
                "figure_number": img.get("figure_number"),
                "size_kb":       round(len(b64) * 3 / 4 / 1024, 1),
            })

        return {
            "total_captions": len(captions),
            "total_images":   len(images),
            "captions": sorted(
                [{"fig": c.get("figure_number"), "page": c.get("page"), "text": c.get("content","")[:80]}
                 for c in captions],
                key=lambda x: int(float(x["fig"] or 0))
            ),
            "images": sorted(
                images,
                key=lambda x: int(float(x.get("figure_number") or 0))
            ),
        }
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str


# ─────────────────────────────────────────────
# INTENT DETECTION
# ─────────────────────────────────────────────
_FIGURE_KW  = {"show","diagram","figure","image","graph","picture",
               "illustration","chart","topology","layout","display","draw","visual"}
_COMPARE_KW = {"compare","difference","vs","versus","contrast",
               "better","faster","advantage","disadvantage"}
_SPEC_KW    = {"spec","specification","how many","how much","memory",
               "bandwidth","tflop","core","watt","gb","tb","ghz",
               "capacity","number of","count","size"}

def detect_intent(query: str) -> str:
    q     = query.lower()
    words = set(q.split())
    if _FIGURE_KW & words or "show me" in q or "show the" in q:
        return "FIGURE"
    if _COMPARE_KW & words:
        return "COMPARE"
    if _SPEC_KW & words or any(k in q for k in _SPEC_KW):
        return "SPEC"
    if any(k in q for k in ["how does","how do","explain","what is","what are","describe","overview"]):
        return "HOW"
    return "GENERAL"


# ─────────────────────────────────────────────
# REDIS HELPERS
# ─────────────────────────────────────────────

def _parse(results) -> list[dict]:
    if not results or results[0] == 0:
        return []
    docs = []
    for i in range(1, len(results), 2):
        fields = results[i + 1]
        d: dict = {}
        for j in range(0, len(fields), 2):
            k = fields[j].decode() if isinstance(fields[j], bytes) else fields[j]
            v = fields[j+1]
            if isinstance(v, bytes):
                try:    v = v.decode()
                except: v = v.decode("latin-1")
            d[k] = v
        docs.append(d)
    return docs


def vector_search_text(query: str, top_k: int = 14) -> list[dict]:
    qvec = np.array(
        model.encode(f"query: {query}", normalize_embeddings=True),
        dtype=np.float32,
    ).tobytes()
    res = r.execute_command(
        "FT.SEARCH", INDEX_NAME,
        "@type:{TEXT|CAPTION}=>[KNN %d @embedding $vec AS vsc]" % top_k,
        "PARAMS", 2, "vec", qvec,
        "SORTBY", "vsc",
        "RETURN", 5, "type","content","page","figure_number","vsc",
        "DIALECT", 2,
    )
    return _parse(res)


def fulltext_captions(query: str, top_k: int = 8) -> list[dict]:
    clean = " ".join(re.sub(r"[^a-zA-Z0-9 ]", " ", query).split())
    if not clean:
        return []
    try:
        res = r.execute_command(
            "FT.SEARCH", INDEX_NAME,
            f"@type:{{CAPTION}} @content:({clean})",
            "LIMIT", 0, top_k,
            "RETURN", 4, "type","content","page","figure_number",
            "DIALECT", 2,
        )
        return _parse(res)
    except Exception:
        return []


def get_all_captions() -> list[dict]:
    try:
        res = r.execute_command(
            "FT.SEARCH", INDEX_NAME,
            "@type:{CAPTION}",
            "LIMIT", 0, 50,
            "RETURN", 3, "content","page","figure_number",
            "DIALECT", 2,
        )
        docs = _parse(res)
        return [d for d in docs if _valid_fig(d.get("figure_number",""))]
    except Exception:
        return []


def _valid_fig(fn) -> bool:
    try:
        return int(float(fn)) > 0
    except (ValueError, TypeError):
        return False


def fetch_images_for_figure(figure_number: int, max_imgs: int = 1) -> list[str]:
    try:
        res = r.execute_command(
            "FT.SEARCH", INDEX_NAME,
            f"@type:{{IMAGE}} @figure_number:[{figure_number} {figure_number}]",
            "LIMIT", 0, 10,
            "RETURN", 1, "content",
            "DIALECT", 2,
        )
        docs = _parse(res)
        print(f"  fetch_images fig={figure_number}: {len(docs)} result(s)")
        docs.sort(key=lambda d: len(d.get("content","")), reverse=True)
        seen: set[str] = set()
        result = []
        for doc in docs:
            b64 = doc.get("content","")
            key = b64[:80]
            if b64 and key not in seen:
                seen.add(key)
                result.append(b64)
                if len(result) >= max_imgs:
                    break
        return result
    except Exception as e:
        print(f"  Image fetch error fig {figure_number}: {e}")
        return []


def fetch_images_by_pages(pages: list[str], max_imgs: int = 1) -> list[str]:
    candidates = []
    seen: set[str] = set()
    for p in pages:
        try:
            pint = int(p)
            res = r.execute_command(
                "FT.SEARCH", INDEX_NAME,
                f"@type:{{IMAGE}} @page:[{pint} {pint}]",
                "LIMIT", 0, 5,
                "RETURN", 1, "content",
                "DIALECT", 2,
            )
            for doc in _parse(res):
                b64 = doc.get("content","")
                key = b64[:80]
                if b64 and key not in seen:
                    seen.add(key)
                    candidates.append(b64)
        except Exception:
            continue
    candidates.sort(key=len, reverse=True)
    return candidates[:max_imgs]


# ─────────────────────────────────────────────
# LLM: PICK BEST FIGURE
# ─────────────────────────────────────────────

def llm_pick_best_figure(query: str, captions: list[dict]) -> int | None:
    if not captions:
        return None

    caption_list = ""
    for c in sorted(captions, key=lambda x: int(float(x.get("figure_number",0)))):
        fn = int(float(c.get("figure_number", 0)))
        caption_list += f"  Figure {fn} (page {c.get('page','?')}): {c.get('content','')}\n"

    prompt = f"""You are helping retrieve the correct diagram from a technical document.

User query: "{query}"

Available figures:
{caption_list}

Which single figure number best matches the user's request?
Reply with ONLY the integer figure number (e.g. 2).
If none match, reply with 0."""

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        raw = resp.choices[0].message.content.strip()
        n = int(re.search(r"\d+", raw).group())
        print(f"  LLM picked figure: {n}")
        return n if n > 0 else None
    except Exception as e:
        print(f"  Figure picker error: {e}")
        return None


# ─────────────────────────────────────────────
# LLM RERANKER
# ─────────────────────────────────────────────

def rerank(query: str, candidates: list[dict], top_n: int = 6) -> list[dict]:
    if not candidates:
        return []

    listing = ""
    for idx, doc in enumerate(candidates):
        listing += (
            f"\nID {idx} | Page {doc.get('page','?')} | {doc.get('type','?')}:\n"
            f"{doc.get('content','')[:350]}\n"
        )

    prompt = f"""You are a relevance judge for a technical document QA system.

Query: "{query}"

Passages:
{listing}

Score each ID 0-10 for relevance to the query.
Return ONLY valid JSON: {{"scores": {{"0": 8, "1": 2, ...}}}}
No explanation."""

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=300,
        )
        raw    = resp.choices[0].message.content.strip()
        jm     = re.search(r"\{.*\}", raw, re.DOTALL)
        if not jm: raise ValueError("No JSON")
        scores = json.loads(jm.group()).get("scores", {})
        ranked = sorted(
            [(int(k), float(v)) for k, v in scores.items()],
            key=lambda x: x[1], reverse=True,
        )
        sel = [candidates[i] for i, s in ranked[:top_n] if s >= 3 and i < len(candidates)]
        return sel if sel else candidates[:top_n]
    except Exception as e:
        print(f"  Reranker error: {e}")
        return candidates[:top_n]


# ─────────────────────────────────────────────
# ANSWER GENERATION
# ─────────────────────────────────────────────

def generate_answer(query: str, context: str, intent: str) -> str:
    style = {
        "SPEC":    "Give precise facts and exact numbers from the context.",
        "COMPARE": "Structure as a clear comparison with specific data points.",
        "HOW":     "Explain step-by-step using only the context.",
        "FIGURE":  "Briefly describe what the figure shows based on its caption.",
        "GENERAL": "Answer clearly and concisely.",
    }.get(intent, "Answer clearly and concisely.")

    prompt = f"""You are a precise technical assistant for NVIDIA DGX-1 documentation.

RULES:
1. Answer ONLY from the Context below.
2. If context is insufficient, say so explicitly.
3. Cite page numbers for specific facts.
4. {style}
5. Be concise.

Context:
{context}

Question: {query}

Answer:"""

    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# MAIN ENDPOINT
# ─────────────────────────────────────────────

@app.post("/multimedia-rag")
def multimedia_rag(request: QueryRequest):
    query = request.query.strip()
    if not query:
        return {"error": "Empty query."}

    intent = detect_intent(query)
    print(f"\nQuery: {query!r}  intent={intent}")

    # Step 1: Retrieve TEXT + CAPTION candidates
    vec_docs = vector_search_text(query, top_k=14)
    cap_docs = fulltext_captions(query, top_k=8)

    seen: set[str] = set()
    candidates: list[dict] = []
    for doc in vec_docs + cap_docs:
        key = doc.get("content","")[:80]
        if key not in seen:
            seen.add(key)
            candidates.append(doc)

    if not candidates:
        return {"error": "No relevant document chunks found. Try rephrasing."}

    # Step 2: Rerank
    selected = rerank(query, candidates[:12], top_n=6)

    # Step 3: Build context
    text_blocks: list[str] = []
    pages: set[str] = set()

    for doc in selected:
        p = str(doc.get("page","?"))
        pages.add(p)
        if doc["type"] == "TEXT":
            text_blocks.append(f"[Page {p}] {doc['content']}")
        elif doc["type"] == "CAPTION":
            fn = doc.get("figure_number","?")
            text_blocks.append(f"[Page {p}] Figure {fn}: {doc['content']}")

    context_text = "\n\n".join(text_blocks)

    # Step 4: Fetch image (FIGURE queries only)
    images: list[str] = []
    best_fig: int | None = None

    if intent == "FIGURE":
        all_captions = get_all_captions()
        best_fig = llm_pick_best_figure(query, all_captions)

        if best_fig:
            images = fetch_images_for_figure(best_fig, max_imgs=1)

        if not images and pages:
            print(f"  Fallback to page search: {pages}")
            images = fetch_images_by_pages(sorted(pages)[:3], max_imgs=1)

    # Step 5: Generate answer
    answer = generate_answer(query, context_text, intent)

    if intent == "FIGURE" and best_fig and images:
        answer = f"Showing Figure {best_fig} from the document.\n\n" + answer

    return {
        "answer":    answer,
        "citations": sorted(pages),
        "images":    images,
    }
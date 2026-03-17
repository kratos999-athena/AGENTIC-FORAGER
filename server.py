"""
Signal Terminal — FastAPI Backend (Autonomous Mode)
====================================================
Endpoints:
  GET  /                    — Serves index.html
  POST /api/pipeline/run    — Manual one-shot (live HN data), returns results directly
  GET  /api/pipeline/state  — GLOBAL_RESULTS + countdown to next background run
  POST /api/chat/stream     — SSE stream for Agent 4
"""

from __future__ import annotations

import asyncio
import json
import os
import requests as req_lib
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from groq import Groq
from pydantic import BaseModel

from agent1_scout import get_mock_ingestion_batch, get_live_hn_batch, run_scout_batch
from agent2_github_quant import EnrichedItem, enrich_signal
from agent3_partner import build_context_block, build_mock_enriched_item, draft_memo

BASE_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# §1  Global state
# ─────────────────────────────────────────────────────────────────────────────

SEEN_HN_IDS:    set[int]   = set()
GLOBAL_RESULTS: list[dict] = []
# BUG FIX 2: initialise to now+1800 so countdown is correct from the first request
NEXT_RUN_AT:    datetime   = datetime.now(timezone.utc) + timedelta(seconds=1800)
PIPELINE_LOCK:  asyncio.Lock | None = None

SCAN_INTERVAL_SECONDS = 1800


# ─────────────────────────────────────────────────────────────────────────────
# §2  Serialisation
# ─────────────────────────────────────────────────────────────────────────────

def _serialise_enriched(ei: EnrichedItem) -> dict:
    m   = ei.primary_metrics
    cv  = m.commit_velocity  if m else None
    ir  = m.issue_resolution if m else None
    pro = m.profile          if m else None

    return {
        "item_id":    ei.classified.raw.item_id,
        "title":      ei.classified.raw.title,
        "url":        ei.classified.raw.url,
        "source":     ei.classified.raw.source.value.upper(),
        "confidence": ei.classified.confidence,
        "rationale": {
            "summary":             ei.classified.rationale.one_line_summary,
            "architectural_shift": ei.classified.rationale.architectural_shift,
            "developer_adoption":  ei.classified.rationale.developer_adoption,
            "oss_milestone":       ei.classified.rationale.oss_milestone,
        },
        "repo": {
            "name":     pro.name_with_owner if pro else "unknown/repo",
            "language": pro.primary_language if pro else None,
            "stars":    pro.stars if pro else None,
            "forks":    pro.forks if pro else None,
            "watchers": pro.watchers if pro else None,
            "archived": pro.is_archived if pro else False,
            "branch":   pro.default_branch if pro else None,
        } if pro else None,
        "velocity": {
            "d30":          cv.last_30_days if cv else None,
            "d60":          cv.last_60_days if cv else None,
            "d90":          cv.last_90_days if cv else None,
            "weekly_avg":   cv.weekly_avg_30d if cv else None,
            "acceleration": cv.acceleration if cv else "INSUFFICIENT",
        } if cv else None,
        "issues": {
            "enabled":      ir.has_issues_enabled if ir else False,
            "sample_size":  ir.sample_size if ir else 0,
            "total_closed": ir.total_closed_issues if ir else None,
            "median_days":  ir.median_days if ir else None,
            "p25_days":     round(ir.p25_hours / 24, 2) if ir and ir.p25_hours else None,
            "p75_days":     round(ir.p75_hours / 24, 2) if ir and ir.p75_hours else None,
            "grade":        ir.responsiveness_grade if ir else "UNKNOWN",
        } if ir else None,
        "data_gaps":           [gap for m in ei.metrics for gap in m.data_gaps],
        "has_sufficient_data": ei.has_sufficient_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# §3  Shared pipeline logic (used by both background loop and manual endpoint)
# ─────────────────────────────────────────────────────────────────────────────

async def _run_agents_on_batch(raw_batch, client, github_token) -> list[dict]:
    """
    Agent 1 → Agent 2 → Agent 3 on a given raw_batch.
    Returns list of result dicts ready for GLOBAL_RESULTS.
    All blocking calls are wrapped in run_in_executor.
    """
    loop = asyncio.get_event_loop()

    # Agent 1
    signals, _ = await loop.run_in_executor(
        None, lambda: run_scout_batch(raw_batch, client, verbose=False)
    )
    if not signals:
        return []

    # Agent 2
    session = req_lib.Session()
    enriched_items: list[EnrichedItem] = []
    for sig in signals:
        ei = await loop.run_in_executor(
            None, lambda s=sig: enrich_signal(s, github_token, session)
        )
        enriched_items.append(ei)
    session.close()

    # Agent 3
    results: list[dict] = []
    for ei in enriched_items:
        if not ei.metrics:
            continue
        try:
            memo_md = await loop.run_in_executor(
                None, lambda e=ei: draft_memo(e, client, verbose=False)
            )
            results.append({
                "meta":       _serialise_enriched(ei),
                "memo_md":    memo_md,
                "context":    build_context_block(ei),
                "discovered": datetime.now(timezone.utc).isoformat(),
                "source":     "autonomous",
            })
        except Exception as exc:
            print(f"[Pipeline] Memo failed for {ei.classified.raw.item_id}: {exc}")

    return results


# async def process_live_pipeline() -> int:
#     """Live HN ingestion run. Updates SEEN_HN_IDS and GLOBAL_RESULTS. Returns new count."""
#     global SEEN_HN_IDS, GLOBAL_RESULTS
#     loop         = asyncio.get_event_loop()
#     client       = Groq()
#     github_token = os.environ.get("GITHUB_TOKEN", "")

#     print(f"[{datetime.now(timezone.utc).isoformat()}] Autonomous pipeline starting…")

#     raw_batch = await loop.run_in_executor(
#         None, lambda: get_live_hn_batch(limit=20, seen_ids=SEEN_HN_IDS)
#     )

#     # Mark all fetched IDs seen before classification to avoid re-fetching on error
#     for item in raw_batch:
#         try:
#             SEEN_HN_IDS.add(int(item.item_id.replace("hn_", "")))
#         except ValueError:
#             pass

#     if not raw_batch:
#         print("[Pipeline] No new HN items.")
#         return 0

#     new_results = await _run_agents_on_batch(raw_batch, client, github_token)
#     GLOBAL_RESULTS = (new_results + GLOBAL_RESULTS)[:100]
#     print(f"[Pipeline] +{len(new_results)} new result(s). Total: {len(GLOBAL_RESULTS)}")
#     return len(new_results)

async def process_live_pipeline() -> int:
    """
    Full autonomous run: HN → Agent 1 → Agent 2 → Agent 3.
    Prepends new results to GLOBAL_RESULTS and updates SEEN_HN_IDS.
    Returns the number of new signals successfully processed.
    """
    global SEEN_HN_IDS, GLOBAL_RESULTS

    loop         = asyncio.get_event_loop()
    client       = Groq()
    github_token = os.environ.get("GITHUB_TOKEN", "")

    print(f"[{datetime.now(timezone.utc).isoformat()}] Autonomous pipeline starting…")

    # ── Agent 1: live HN ingestion ────────────────────────────────────────
    # get_live_hn_batch now mutates seen_ids in-place for no-GitHub stories,
    # so pass SEEN_HN_IDS directly rather than a copy.
    raw_batch = await loop.run_in_executor(
        None, lambda: get_live_hn_batch(limit=10, seen_ids=SEEN_HN_IDS, min_hn_score=40)
    )

    # Mark all fetched items as seen regardless of downstream classification
    for item in raw_batch:
        try:
            SEEN_HN_IDS.add(int(item.item_id.replace("hn_", "")))
        except ValueError:
            pass

    if not raw_batch:
        print("[Pipeline] No new HN items with GitHub URLs found.")
        return 0

    signals, _ = await loop.run_in_executor(
        None, lambda: run_scout_batch(raw_batch, client, verbose=False)
    )

    if not signals:
        print("[Pipeline] No SIGNAL items after Scout classification.")
        return 0

    print(f"[Pipeline] {len(signals)} signal(s) from Agent 1.")

    # ── Agent 2: GitHub enrichment ────────────────────────────────────────
    session = req_lib.Session()
    enriched_items: list[EnrichedItem] = []

    for sig in signals:
        ei = await loop.run_in_executor(
            None, lambda s=sig: enrich_signal(s, github_token, session)
        )
        enriched_items.append(ei)

    session.close()

    # ── Agent 3: memo drafting (with star-count quality gate) ─────────────
    MIN_STARS = 100

    new_results: list[dict] = []
    for ei in enriched_items:
        if not ei.metrics:
            continue

        # Star-count quality gate — skip tiny / personal projects before
        # spending any LLM tokens on memo generation
        repo_name = ei.primary_metrics.profile.name_with_owner if ei.primary_metrics else "unknown"
        stars     = ei.primary_metrics.profile.stars if ei.primary_metrics else None

        if stars is None:
            # No star data means Agent 2 couldn't fetch metrics (likely no token).
            # Let it through rather than silently dropping it.
            print(f"[Pipeline] {repo_name} — star count unavailable, proceeding.")
        elif stars < MIN_STARS:
            print(f"[Pipeline] Skipping {repo_name} — only {stars} star(s) (threshold: {MIN_STARS}).")
            continue

        try:
            memo_md = await loop.run_in_executor(
                None, lambda e=ei: draft_memo(e, client, verbose=False)
            )
            new_results.append({
                "meta":       _serialise_enriched(ei),
                "memo_md":    memo_md,
                "context":    build_context_block(ei),
                "discovered": datetime.now(timezone.utc).isoformat(),
                "source":     "autonomous",
            })
        except Exception as exc:
            print(f"[Pipeline] Memo failed for {ei.classified.raw.item_id}: {exc}")

    # Prepend newest results (keep cap at 100 to avoid unbounded memory)
    GLOBAL_RESULTS = (new_results + GLOBAL_RESULTS)[:100]

    print(f"[Pipeline] +{len(new_results)} new result(s). Total: {len(GLOBAL_RESULTS)}")
    return len(new_results)


# ─────────────────────────────────────────────────────────────────────────────
# §4  Background loop
# ─────────────────────────────────────────────────────────────────────────────

async def _background_loop():
    global NEXT_RUN_AT
    await asyncio.sleep(10)   # let server finish starting up
    while True:
        # BUG FIX 3: set NEXT_RUN_AT BEFORE running so countdown is correct
        # during the (potentially multi-minute) pipeline execution
        NEXT_RUN_AT = datetime.now(timezone.utc) + timedelta(seconds=SCAN_INTERVAL_SECONDS)
        async with PIPELINE_LOCK:
            try:
                await process_live_pipeline()
            except Exception as exc:
                print(f"[BG loop error] {exc}")
        print(f"[Pipeline] Next autonomous run at {NEXT_RUN_AT.isoformat()}")
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


# ─────────────────────────────────────────────────────────────────────────────
# §5  App + lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global PIPELINE_LOCK
    PIPELINE_LOCK = asyncio.Lock()
    task = asyncio.create_task(_background_loop())
    print("[Startup] Autonomous background pipeline scheduled (every 30 min).")
    yield
    task.cancel()


app = FastAPI(title="Signal Terminal", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# §6  /api/pipeline/state
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/pipeline/state")
async def pipeline_state():
    now  = datetime.now(timezone.utc)
    secs = max(0, int((NEXT_RUN_AT - now).total_seconds()))
    return {
        "results":                GLOBAL_RESULTS,
        "total":                  len(GLOBAL_RESULTS),
        "next_run_at":            NEXT_RUN_AT.isoformat(),
        "seconds_until_next_run": secs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# §7  /api/pipeline/run  (manual trigger — live HN data)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/pipeline/run")
async def run_pipeline_manual():
    """
    Manual trigger: same live HN pipeline as the background loop.

    BUG FIX 1: Returns {results, total, ran_at} so the frontend can
    populate state.results directly from this response, instead of
    returning only {message, ran_at} which caused an empty feed.
    """
    global GLOBAL_RESULTS
    loop         = asyncio.get_event_loop()
    client       = Groq()
    github_token = os.environ.get("GITHUB_TOKEN", "")

    raw_batch = await loop.run_in_executor(
        None, lambda: get_live_hn_batch(limit=20, seen_ids=SEEN_HN_IDS)
    )

    # Mark seen before processing
    for item in raw_batch:
        try:
            SEEN_HN_IDS.add(int(item.item_id.replace("hn_", "")))
        except ValueError:
            pass

    if not raw_batch:
        # Fallback to mock data so the UI always gets something on first run
        raw_batch = get_mock_ingestion_batch()
        new_results = await _run_agents_on_batch(raw_batch, client, github_token)
        for r in new_results:
            r["source"] = "mock"
    else:
        new_results = await _run_agents_on_batch(raw_batch, client, github_token)

    # Merge into global (deduplicate by item_id)
    existing_ids = {r["meta"]["item_id"] for r in GLOBAL_RESULTS}
    new_only     = [r for r in new_results if r["meta"]["item_id"] not in existing_ids]
    GLOBAL_RESULTS = (new_only + GLOBAL_RESULTS)[:100]

    return {
        "results": GLOBAL_RESULTS,   # full list so frontend stays in sync
        "total":   len(GLOBAL_RESULTS),
        "ran_at":  datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# §8  /api/chat/stream  — Agent 4 SSE
# ─────────────────────────────────────────────────────────────────────────────

AGENT4_SYSTEM = """
You are Agent 4: The Research Assistant — a specialist analyst inside a VC research terminal.
Answer questions ONLY about the project in your context. Rules:
1. Every quantitative claim must cite a metric from QUANTITATIVE CONTEXT.
2. If data is in DATA GAPS, quote the gap string exactly and state it is unavailable.
3. No external knowledge. No invented numbers. No hedged qualitative language.
4. Be concise and technical. Your reader has already read the memo.

QUANTITATIVE CONTEXT
====================
{context}

INVESTMENT MEMO
===============
{memo}
""".strip()


class ChatRequest(BaseModel):
    context:  str
    memo_md:  str
    messages: list[dict]
    repo:     str


async def _sse_stream(request: ChatRequest) -> AsyncGenerator[str, None]:
    """
    BUG FIX 4: The previous version called client.chat.completions.create()
    (a blocking synchronous call) directly inside an async function, which
    blocked the entire event loop during streaming.

    Fixed by collecting all tokens in run_in_executor (thread pool) and
    then yielding them. For true token-by-token streaming, a proper async
    Groq client would be needed; this approach batches in thread then streams
    to the browser which gives the same UX with correct async behaviour.
    """
    client = Groq()
    system = AGENT4_SYSTEM.format(context=request.context, memo=request.memo_md)
    messages = [{"role": "system", "content": system}] + request.messages

    try:
        loop = asyncio.get_event_loop()

        def _collect_tokens() -> list[str]:
            tokens = []
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=1024,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    tokens.append(delta)
            return tokens

        tokens = await loop.run_in_executor(None, _collect_tokens)
        for token in tokens:
            yield f"data: {json.dumps({'token': token})}\n\n"

    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        _sse_stream(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# §9  Serve frontend
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTMLResponse((BASE_DIR / "index.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    
    # BUG FIX 5: reload=True resets all global state on every file save,
    # killing the background task and clearing SEEN_HN_IDS / GLOBAL_RESULTS.
    # Use reload=False in production. Use reload=True only if you accept that
    # global state will reset on each reload.
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
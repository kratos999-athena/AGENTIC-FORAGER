"""
Phase 1 — Agent 1: The Scout
=============================
Responsibilities:
  1. Ingest raw items from data sources (HackerNews, RSS, Twitter).
  2. Classify each item as SIGNAL or NOISE via a structured Groq API call.
  3. Emit typed, validated RawItem → ClassifiedItem objects downstream.

Architecture note: This module is intentionally stateless. The Scout receives
a batch of RawItems and returns a list of ClassifiedItems. Orchestration
(scheduling, queueing) lives outside this file.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from groq import Groq
from pydantic import BaseModel, Field, HttpUrl, field_validator


# ─────────────────────────────────────────────
# §1  Enumerations
# ─────────────────────────────────────────────

class DataSource(str, Enum):
    """Supported ingestion sources."""
    HACKERNEWS = "hackernews"
    RSS        = "rss"
    TWITTER    = "twitter"
    MANUAL     = "manual"          # direct URL / paste for ad-hoc analysis


class Classification(str, Enum):
    """Strict binary verdict produced by Agent 1."""
    SIGNAL = "SIGNAL"
    NOISE  = "NOISE"


# ─────────────────────────────────────────────
# §2  Data Models
# ─────────────────────────────────────────────

class RawItem(BaseModel):
    """
    A single unprocessed item arriving from any ingestion source.
    This is the *input* contract for Agent 1.
    """
    item_id:      str                  = Field(...,  description="Stable unique ID from the source (e.g. HN story ID).")
    source:       DataSource           = Field(...,  description="Which data source produced this item.")
    title:        str                  = Field(...,  description="Headline or tweet text (≤ 280 chars recommended).")
    body:         Optional[str]        = Field(None, description="Full post body, article snippet, or thread text.")
    url:          Optional[str]        = Field(None, description="Primary URL associated with the item.")
    github_urls:  list[str]            = Field(default_factory=list,
                                               description="Any GitHub repo URLs extracted from title+body.")
    ingested_at:  datetime             = Field(default_factory=lambda: datetime.now(timezone.utc),
                                               description="UTC timestamp of ingestion.")
    raw_score:    Optional[int]        = Field(None, description="Source-native popularity score (HN points, RT count, etc.).")

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("title must not be blank")
        return v.strip()


class ScoutRationale(BaseModel):
    """
    Structured reasoning block embedded in every ClassifiedItem.
    Forces the LLM to justify its verdict with explicit criteria checks
    before a classification is accepted — our first hallucination guard.
    """
    architectural_shift:  bool = Field(..., description="Does this represent a fundamental architectural improvement?")
    developer_adoption:   bool = Field(..., description="Is there evidence of high or rapidly growing developer adoption?")
    oss_milestone:        bool = Field(..., description="Is this a significant open-source release or contribution milestone?")
    noise_signal:         bool = Field(..., description="Is the item primarily marketing, funding news, or a minor patch?")
    one_line_summary:     str  = Field(..., description="Single sentence capturing the core engineering claim.")


class ClassifiedItem(BaseModel):
    """
    The *output* contract of Agent 1.
    Consumed by Agent 2 (GitHub Quant) or dropped if NOISE.
    """
    raw:             RawItem          = Field(..., description="Original ingested item, unchanged.")
    classification:  Classification   = Field(..., description="SIGNAL or NOISE verdict.")
    confidence:      float            = Field(..., ge=0.0, le=1.0,
                                              description="Scout's self-reported confidence (0–1).")
    rationale:       ScoutRationale   = Field(..., description="Structured reasoning that supports the verdict.")
    classified_at:   datetime         = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_signal(self) -> bool:
        return self.classification == Classification.SIGNAL

    @property
    def has_github_targets(self) -> bool:
        return len(self.raw.github_urls) > 0


# ─────────────────────────────────────────────
# §3  System Prompt
# ─────────────────────────────────────────────

SCOUT_SYSTEM_PROMPT = """
You are The Scout — the first agent in a ruthless technical venture-capital analysis pipeline.

## Your Mission
Classify a raw tech item as SIGNAL or NOISE based solely on its engineering merit.
You are allergic to hype. You care only about:
  - Fundamental architectural shifts (new paradigm, not incremental tweak)
  - Developer momentum (adoption velocity, community growth, contributor count)
  - Significant open-source milestones (major version, breakthrough benchmark, pivotal contributor joining)

## Hard Discard Rules — Auto-NOISE if ANY of these are true
  - The item is primarily about a funding round, valuation, or acquisition
  - The item is marketing copy with no technical substance
  - The item describes a minor patch (< semver minor bump) with no novel architecture
  - The item is speculative price/token analysis
  - The item is a job posting or conference announcement

## Output Format — STRICT JSON, NO MARKDOWN FENCES
Return exactly this JSON object and nothing else:
{
  "classification": "SIGNAL" | "NOISE",
  "confidence": <float 0.0-1.0>,
  "rationale": {
    "architectural_shift": <bool>,
    "developer_adoption": <bool>,
    "oss_milestone": <bool>,
    "noise_signal": <bool>,
    "one_line_summary": "<single sentence>"
  }
}

Be ruthless. When in doubt, classify as NOISE.
""".strip()


# ─────────────────────────────────────────────
# §4  Core Classification Logic
# ─────────────────────────────────────────────

def _build_user_message(item: RawItem) -> str:
    """Serialize a RawItem into the plaintext payload sent to the Scout LLM."""
    parts = [
        f"SOURCE: {item.source.value.upper()}",
        f"TITLE: {item.title}",
    ]
    if item.body:
        truncated_body = item.body[:1200] + ("..." if len(item.body) > 1200 else "")
        parts.append(f"BODY: {truncated_body}")
    if item.url:
        parts.append(f"URL: {item.url}")
    if item.github_urls:
        parts.append(f"GITHUB REPOS: {', '.join(item.github_urls)}")
    if item.raw_score is not None:
        parts.append(f"POPULARITY SCORE: {item.raw_score}")
    return "\n".join(parts)


def classify_item(item: RawItem, client: Groq) -> ClassifiedItem:
    """
    Send a single RawItem to the Scout LLM and return a validated ClassifiedItem.

    Raises:
        ValueError: If the LLM returns malformed JSON or fails schema validation.
        groq.APIError: On network / rate-limit failures (caller should retry).
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=512,
        messages=[
            {"role": "system", "content": SCOUT_SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_message(item)},
        ],
    )

    raw_text = response.choices[0].message.content.strip()

    # Strip accidental markdown fences if the model slips up
    raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text)
    raw_text = re.sub(r"\n?```$", "", raw_text)

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Scout returned non-JSON output for item '{item.item_id}': {raw_text}") from exc

    rationale = ScoutRationale(**payload["rationale"])

    return ClassifiedItem(
        raw=item,
        classification=Classification(payload["classification"]),
        confidence=float(payload["confidence"]),
        rationale=rationale,
    )


def run_scout_batch(
    items: list[RawItem],
    client: Groq,
    *,
    verbose: bool = True,
) -> tuple[list[ClassifiedItem], list[ClassifiedItem]]:
    """
    Classify a batch of RawItems.

    Returns:
        (signals, noise) — two separate lists for downstream routing.
    """
    signals: list[ClassifiedItem] = []
    noise:   list[ClassifiedItem] = []

    for item in items:
        if verbose:
            print(f"  -> Classifying [{item.item_id}] \"{item.title[:60]}...\"")
        try:
            result = classify_item(item, client)
            bucket = signals if result.is_signal else noise
            bucket.append(result)
            if verbose:
                icon = "SIGNAL" if result.is_signal else "NOISE"
                print(f"    {icon}  (confidence={result.confidence:.2f}) -- {result.rationale.one_line_summary}")
        except (ValueError, KeyError) as exc:
            print(f"    WARNING: Classification failed for {item.item_id}: {exc}")

    return signals, noise


# ─────────────────────────────────────────────
# §5  Mock Ingestion Data  (Phase 1 testing)
# ─────────────────────────────────────────────

def get_mock_ingestion_batch() -> list[RawItem]:
    """
    Returns a curated set of 8 realistic items that stress-test the
    Signal/Noise boundary -- 4 expected SIGNALs, 4 expected NOISE.
    """
    return [
        # -- Expected SIGNAL --
        RawItem(
            item_id="hn_41892345",
            source=DataSource.HACKERNEWS,
            title="Mojo 1.0 released: Python-compatible language hits C-speed via MLIR, 35k GitHub stars in 30 days",
            body=(
                "Modular has shipped Mojo 1.0, a Python-superset that compiles via MLIR to achieve "
                "near-C performance with full CPython interop. The repo crossed 35k stars in under a "
                "month and has 420 contributors. Key architectural win: unified compiler IR that allows "
                "progressive typing without rewriting existing Python codebases."
            ),
            url="https://www.modular.com/blog/mojo-1-0",
            github_urls=["https://github.com/modularml/mojo"],
            raw_score=2841,
        ),
        RawItem(
            item_id="hn_41903210",
            source=DataSource.HACKERNEWS,
            title="Deno 2.0: Node.js compatibility layer complete, npm package support ships with zero-config",
            body=(
                "Deno 2.0 ships full npm compatibility via a new resolver that maps Node built-ins to "
                "Deno's secure runtime. No package.json required. Early benchmarks show 2.1x faster cold "
                "start vs Node 22 for serverless workloads. 94k GitHub stars, 1,800 contributors."
            ),
            url="https://deno.com/blog/v2",
            github_urls=["https://github.com/denoland/deno"],
            raw_score=1923,
        ),
        RawItem(
            item_id="rss_llvm_weekly_88",
            source=DataSource.RSS,
            title="LLVM merges Clang-based Carbon interop layer -- bidirectional C++ <-> Carbon FFI at zero overhead",
            body=(
                "The Carbon Language project's long-awaited C++ interop PR was merged into the main LLVM "
                "monorepo. The implementation uses a novel 'bridge translation unit' approach that allows "
                "calling Carbon from C++ and vice versa with no runtime indirection. This unblocks migration "
                "paths for large C++ codebases without a full rewrite."
            ),
            url="https://github.com/carbon-language/carbon-lang/pull/3812",
            github_urls=["https://github.com/carbon-language/carbon-lang"],
            raw_score=None,
        ),
        RawItem(
            item_id="tw_ThePSF_2910",
            source=DataSource.TWITTER,
            title="uv 0.4 ships: Rust-based pip/virtualenv replacement, 100x faster resolver, now handles workspaces",
            body=(
                "Astral's uv 0.4 adds PEP 723 inline script metadata, multi-project workspaces, and a "
                "lockfile format. Resolver benchmarks: Django + 220 deps resolves in 38ms vs pip's 4.1s. "
                "38k stars, merging 30+ PRs/week. Effectively replacing pip in modern Python toolchains."
            ),
            url="https://astral.sh/blog/uv-0-4",
            github_urls=["https://github.com/astral-sh/uv"],
            raw_score=4102,
        ),

        # -- Expected NOISE --
        RawItem(
            item_id="hn_41887654",
            source=DataSource.HACKERNEWS,
            title="Acme AI raises $200M Series C at $2B valuation to 'revolutionize enterprise AI'",
            body=(
                "Acme AI, a startup focused on enterprise AI solutions, has raised $200M in a Series C "
                "funding round led by Sequoia Capital, valuing the company at $2 billion. The company "
                "plans to use the funds to expand its sales team and marketing efforts."
            ),
            url="https://techcrunch.com/acme-ai-series-c",
            github_urls=[],
            raw_score=312,
        ),
        RawItem(
            item_id="rss_coindesk_771",
            source=DataSource.RSS,
            title="ETH price analysis: on-chain metrics suggest $10k by Q2 2025 according to analyst",
            body=(
                "Ethereum's on-chain metrics are flashing bullish signals, with a prominent crypto analyst "
                "predicting ETH could reach $10,000 by Q2 2025. The analysis cites whale accumulation "
                "patterns and upcoming ETF approvals as key catalysts."
            ),
            url="https://coindesk.com/eth-price-10k-prediction",
            github_urls=[],
            raw_score=89,
        ),
        RawItem(
            item_id="hn_41891122",
            source=DataSource.HACKERNEWS,
            title="React 18.2.1 patch release: fixes minor hydration edge case in StrictMode",
            body=(
                "The React team has released version 18.2.1, a patch that addresses a rare hydration "
                "mismatch warning that could appear in StrictMode when using certain third-party libraries. "
                "No API changes. Users are encouraged to upgrade at their convenience."
            ),
            url="https://github.com/facebook/react/releases/tag/v18.2.1",
            github_urls=["https://github.com/facebook/react"],
            raw_score=156,
        ),
        RawItem(
            item_id="tw_DevConf_0042",
            source=DataSource.TWITTER,
            title="Join us at CloudWorld 2025! Keynotes from 50+ industry leaders. Use code CLOUD25 for 20% off.",
            body=None,
            url="https://cloudworld.example.com/register",
            github_urls=[],
            raw_score=12,
        ),
    ]


# ─────────────────────────────────────────────
# §6  Live HackerNews Ingestion
# ─────────────────────────────────────────────

HN_TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL        = "https://hacker-news.firebaseio.com/v0/item/{}.json"

# Regex: captures github.com/OWNER/REPO — stops at slash, dot, query, space, quote
_GH_RE = re.compile(r"https?://github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+?)(?:[/?#\s\"'<]|$)")


def _hn_get(url: str, timeout: int = 8) -> dict | list | None:
    """Minimal HTTP GET using stdlib only — returns parsed JSON or None on error."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _extract_github_urls(text: str) -> list[str]:
    """
    Pull all unique github.com/owner/repo URLs from an arbitrary text blob.
    Strips trailing .git and normalises to https://github.com/owner/repo.
    """
    if not text:
        return []
    seen, result = set(), []
    for m in _GH_RE.finditer(text):
        path = m.group(1).rstrip(".").rstrip("/")
        # Must have exactly one slash (owner/repo), ignore deep paths
        if path.count("/") == 1:
            canonical = f"https://github.com/{path}"
            if canonical not in seen:
                seen.add(canonical)
                result.append(canonical)
    return result


# def get_live_hn_batch(
#     limit: int = 15,
#     seen_ids: set | None = None,
# ) -> list[RawItem]:
#     """
#     Fetch the current HackerNews top-stories list, skip any IDs already in
#     `seen_ids`, and return up to `limit` valid RawItems.

#     Each item is built from the HN story object:
#       - title   ← story["title"]
#       - body    ← story["text"] (HTML stripped) when present
#       - url     ← story["url"]
#       - github_urls ← extracted from url + text fields combined
#       - raw_score   ← story["score"]

#     Network errors on individual items are silently skipped so a single
#     flaky endpoint can't abort the whole batch.
#     """
#     if seen_ids is None:
#         seen_ids = set()

#     top_ids = _hn_get(HN_TOP_STORIES_URL)
#     if not isinstance(top_ids, list):
#         print("[HN] Failed to fetch top stories list — returning empty batch.")
#         return []

#     items: list[RawItem] = []
#     for story_id in top_ids:
#         if len(items) >= limit:
#             break
#         if story_id in seen_ids:
#             continue

#         story = _hn_get(HN_ITEM_URL.format(story_id))
#         if not story or story.get("type") != "story":
#             continue

#         title = (story.get("title") or "").strip()
#         if not title:
#             continue

#         # Strip basic HTML tags from the optional HN "text" field
#         raw_text = story.get("text") or ""
#         body_clean = re.sub(r"<[^>]+>", " ", raw_text).strip() or None

#         url = story.get("url") or None

#         # Combine url + body text for GitHub URL extraction
#         combined = " ".join(filter(None, [url or "", raw_text]))
#         github_urls = _extract_github_urls(combined)

#         try:
#             item = RawItem(
#                 item_id=f"hn_{story_id}",
#                 source=DataSource.HACKERNEWS,
#                 title=title,
#                 body=body_clean,
#                 url=url,
#                 github_urls=github_urls,
#                 raw_score=story.get("score"),
#             )
#             items.append(item)
#         except Exception:
#             # Pydantic validation failure (e.g. blank title edge case) — skip
#             continue

#     return items

def get_live_hn_batch(
    limit: int = 10,
    seen_ids: set | None = None,
    min_hn_score: int = 40,
) -> list[RawItem]:
    """
    Fetch HN top stories and return up to `limit` items that CONTAIN a GitHub URL.

    Key changes from the previous version:
      - The loop continues past `limit` top_ids candidates until it has
        collected `limit` items that actually contain GitHub URLs. A story
        with no GitHub URL is marked seen and skipped, so we never waste
        a future fetch on it.
      - Stories with a score below `min_hn_score` are skipped immediately
        (before a full item fetch) to filter low-signal noise.
      - A hard cap of 500 API calls prevents an infinite loop if the
        entire top-stories list has fewer than `limit` GitHub stories.
    """
    if seen_ids is None:
        seen_ids = set()

    top_ids = _hn_get(HN_TOP_STORIES_URL)
    if not isinstance(top_ids, list):
        print("[HN] Failed to fetch top stories list — returning empty batch.")
        return []

    items: list[RawItem] = []
    api_calls = 0
    MAX_API_CALLS = 500   # hard safety cap

    for story_id in top_ids:
        # Stop once we've collected enough GitHub-bearing items
        if len(items) >= limit:
            break

        # Safety cap — avoids hammering the API if HN is sparse
        if api_calls >= MAX_API_CALLS:
            print(f"[HN] Safety cap of {MAX_API_CALLS} API calls reached.")
            break

        # Skip already-processed IDs
        if story_id in seen_ids:
            continue

        story = _hn_get(HN_ITEM_URL.format(story_id))
        api_calls += 1

        if not story or story.get("type") != "story":
            seen_ids.add(story_id)
            continue

        # Score gate — skip low-engagement stories before doing any further work
        score = story.get("score") or 0
        if score < min_hn_score:
            seen_ids.add(story_id)
            continue

        title = (story.get("title") or "").strip()
        if not title:
            seen_ids.add(story_id)
            continue

        # Extract GitHub URLs from both the story URL and the body text
        raw_text = story.get("text") or ""
        url      = story.get("url") or None
        combined = " ".join(filter(None, [url or "", raw_text]))
        github_urls = _extract_github_urls(combined)

        # No GitHub URL — mark seen so we never fetch this story again, then skip
        if not github_urls:
            seen_ids.add(story_id)
            continue

        # Strip HTML from body text for the RawItem
        body_clean = re.sub(r"<[^>]+>", " ", raw_text).strip() or None

        try:
            item = RawItem(
                item_id=f"hn_{story_id}",
                source=DataSource.HACKERNEWS,
                title=title,
                body=body_clean,
                url=url,
                github_urls=github_urls,
                raw_score=score,
            )
            items.append(item)
        except Exception:
            # Pydantic validation failure — mark seen and skip
            seen_ids.add(story_id)
            continue

    print(f"[HN] Fetched {len(items)} GitHub-bearing items from {api_calls} API calls.")
    return items


# ─────────────────────────────────────────────
# §7  Entrypoint
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Agent 1: The Scout -- Phase 1 Test Run")
    print("=" * 60)

    client = Groq()  # reads GROQ_API_KEY from env

    batch = get_mock_ingestion_batch()
    print(f"\nIngesting {len(batch)} items from mock data sources...\n")

    signals, noise = run_scout_batch(batch, client, verbose=True)

    print("\n" + "=" * 60)
    print(f"  RESULTS:  {len(signals)} SIGNAL(s)  |  {len(noise)} NOISE item(s)")
    print("=" * 60)

    if signals:
        print("\n-- SIGNALS PASSED TO AGENT 2 --")
        for item in signals:
            github_note = f"  GitHub: {item.raw.github_urls}" if item.has_github_targets else "  (no GitHub URL)"
            print(f"  * [{item.raw.item_id}] {item.raw.title[:70]}")
            print(f"    Confidence: {item.confidence:.2f} | {item.rationale.one_line_summary}")
            print(github_note)

    print("\nPhase 1 complete. Signals ready for Agent 2 (GitHub Quant).\n")
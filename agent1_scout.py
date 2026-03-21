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



class DataSource(str, Enum):
    """Supported ingestion sources."""
    HACKERNEWS = "hackernews"
    RSS        = "rss"
    TWITTER    = "twitter"
    MANUAL     = "manual"          


class Classification(str, Enum):
    SIGNAL = "SIGNAL"
    NOISE  = "NOISE"



class RawItem(BaseModel):
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
  
    chain_of_thought:     str  = Field(..., description="2-3 sentence internal monologue analysing engineering merit before committing to the boolean flags. Must reference specific technical details from the item.")
    architectural_shift:  bool = Field(..., description="Does this represent a fundamental architectural improvement?")
    developer_adoption:   bool = Field(..., description="Is there evidence of high or rapidly growing developer adoption?")
    oss_milestone:        bool = Field(..., description="Is this a significant open-source release or contribution milestone?")
    noise_signal:         bool = Field(..., description="Is the item primarily marketing, funding news, or a minor patch?")
    one_line_summary:     str  = Field(..., description="Single sentence capturing the core engineering claim.")


class ClassifiedItem(BaseModel):
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

## Chain-of-Thought Requirement
Before committing to any boolean flag, you MUST write your internal reasoning in the
`chain_of_thought` field. 2-3 sentences. Reference specific technical claims from the
item text. This reasoning trace is audited — if the booleans don't follow logically
from the CoT, the classification is rejected.

## Output Format — YOU MUST RESPOND WITH A JSON OBJECT
You MUST return a single JSON object and absolutely nothing else — no markdown
fences, no prose, no commentary before or after the JSON.

The JSON object must have exactly this shape:
{
  "classification": "SIGNAL" | "NOISE",
  "confidence": <float 0.0-1.0>,
  "rationale": {
    "chain_of_thought": "<2-3 sentences reasoning through engineering merit>",
    "architectural_shift": <bool>,
    "developer_adoption": <bool>,
    "oss_milestone": <bool>,
    "noise_signal": <bool>,
    "one_line_summary": "<single sentence>"
  }
}

Be ruthless. When in doubt, classify as NOISE.
""".strip()



def _build_user_message(item: RawItem) -> str:
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
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,          
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SCOUT_SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_message(item)},
        ],
    )
    raw_text = response.choices[0].message.content.strip()

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





HN_TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL        = "https://hacker-news.firebaseio.com/v0/item/{}.json"
_GH_RE = re.compile(r"https?://github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+?)(?:[/?#\s\"'<]|$)")


def _hn_get(url: str, timeout: int = 8) -> dict | list | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _extract_github_urls(text: str) -> list[str]:
    if not text:
        return []
    seen, result = set(), []
    for m in _GH_RE.finditer(text):
        path = m.group(1).rstrip(".").rstrip("/")
        if path.count("/") == 1:
            canonical = f"https://github.com/{path}"
            if canonical not in seen:
                seen.add(canonical)
                result.append(canonical)
    return result




def get_live_hn_batch(
    limit: int = 10,
    seen_ids: set | None = None,
    min_hn_score: int = 40,
) -> list[RawItem]:
    if seen_ids is None:
        seen_ids = set()

    top_ids = _hn_get(HN_TOP_STORIES_URL)
    if not isinstance(top_ids, list):
        print("[HN] Failed to fetch top stories list — returning empty batch.")
        return []

    items: list[RawItem] = []
    api_calls = 0
    MAX_API_CALLS = 500   

    for story_id in top_ids:
        if len(items) >= limit:
            break
        if api_calls >= MAX_API_CALLS:
            print(f"[HN] Safety cap of {MAX_API_CALLS} API calls reached.")
            break
        if story_id in seen_ids:
            continue

        story = _hn_get(HN_ITEM_URL.format(story_id))
        api_calls += 1

        if not story or story.get("type") != "story":
            seen_ids.add(story_id)
            continue
        score = story.get("score") or 0
        if score < min_hn_score:
            seen_ids.add(story_id)
            continue

        title = (story.get("title") or "").strip()
        if not title:
            seen_ids.add(story_id)
            continue
        raw_text = story.get("text") or ""
        url      = story.get("url") or None
        combined = " ".join(filter(None, [url or "", raw_text]))
        github_urls = _extract_github_urls(combined)
        if not github_urls:
            seen_ids.add(story_id)
            continue
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
            seen_ids.add(story_id)
            continue

    print(f"[HN] Fetched {len(items)} GitHub-bearing items from {api_calls} API calls.")
    return items


if __name__ == "__main__":
    print("=" * 60)
    print("  Agent 1: The Scout -- Phase 1 Test Run")
    print("=" * 60)

    client = Groq() 

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
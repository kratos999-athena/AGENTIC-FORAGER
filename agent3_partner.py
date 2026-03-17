"""
Phase 3 -- Agent 3: The Partner
================================
Responsibilities:
  1. Accept a fully enriched EnrichedItem from Agent 2.
  2. Serialize its quantitative and qualitative payload into a grounded
     context block that leaves the LLM no room to hallucinate metrics.
  3. Call the Groq API with a strict VC-Partner system prompt.
  4. Return a structured, citation-enforced Markdown investment memo.

Hallucination guardrails (mirroring claude.md section 4):
  - Every claim in the memo must reference its source metric inline.
  - If `data_gaps` is non-empty, a "Data Limitations" section is MANDATORY
    and must reproduce each gap string verbatim -- no paraphrasing.
  - If metrics are entirely absent, the function raises rather than drafting
    a memo based purely on qualitative signal. Guessing is not permitted.
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime, timedelta, timezone
from typing import Optional

from groq import Groq

# -- Local imports from Phase 1 & 2 --
from agent1_scout import (
    ClassifiedItem,
    Classification,
    DataSource,
    RawItem,
    ScoutRationale,
)
from agent2_github_quant import (
    CommitVelocity,
    EnrichedItem,
    IssueResolutionMetrics,
    QuantitativeMetrics,
    RateLimitSnapshot,
    RepoProfile,
)


# ─────────────────────────────────────────────
# §1  System Prompt
# ─────────────────────────────────────────────

PARTNER_SYSTEM_PROMPT = """
You are The Partner -- the final decision-making agent in a ruthless technical venture-capital analysis pipeline. Your output is a formal investment memo that will be read by a Managing Partner before a capital allocation decision.

ROLE & MINDSET
==============
You are a deeply technical VC Partner. You have built compilers, shipped distributed systems, and led open-source communities. You do not care about token prices, marketing copy, LinkedIn posts, or funding announcements. You care exclusively about:
  1. Paradigm-level engineering shifts.
  2. Developer adoption velocity evidenced by code, not claims.
  3. Maintainer execution quality measured by repository health metrics.

Your writing is precise, dense, and opinionated. You do not hedge unless the data forces you to. You do not use filler phrases ("it is worth noting", "this is an exciting development"). Every sentence either asserts a grounded claim or flags a verified uncertainty.

MANDATORY OUTPUT FORMAT -- STRICT COMPLIANCE REQUIRED
=====================================================
You MUST produce exactly the following Markdown sections in exactly this order.
Do not add sections. Do not remove sections. Do not rename sections.

---

# Investment Memo: {REPO_NAME}
**Classification:** SIGNAL -- High Conviction
**Analyst:** The Partner (Autonomous Agent Pipeline)
**Date:** {DATE}
**Source Signal:** {SOURCE}

---

## 1. Thesis / Core Signal
[Synthesize the engineering thesis from the Scout's rationale. State the paradigm shift in one declarative sentence. Explain *why* this is a structural change and not an incremental improvement. Be specific: name the architectural mechanism, the protocol, the compiler optimization, the abstraction boundary. Maximum 150 words.]

## 2. Developer Momentum
[This section MUST be grounded exclusively in the commit velocity data provided in the QUANTITATIVE CONTEXT block. Do not infer or estimate. Required inline citations -- you must include ALL of these verbatim:]
  - Commits (last 30 days): [METRIC]
  - Commits (last 60 days): [METRIC]
  - Commits (last 90 days): [METRIC]
  - Weekly average (30d window): [METRIC] commits/week
  - Momentum trajectory: [ACCELERATING | DECELERATING | STEADY | INSUFFICIENT]

[After the citation block, write 2-3 sentences interpreting the trajectory. If trajectory is INSUFFICIENT due to missing data, you MUST state: "Insufficient quantitative data to evaluate commit momentum." and stop.]

## 3. Maintainer Responsiveness
[This section MUST be grounded exclusively in the issue resolution data provided. Required inline citations:]
  - Median time-to-resolution: [METRIC] days (grade: [A/B/C/D])
  - Sample size: [N] closed issues analysed
  - IQR: P25 = [X] days -- P75 = [Y] days
  - Total closed issues (lifetime): [METRIC]

[After the citation block, write 2-3 sentences interpreting what this responsiveness profile signals about the core team's execution culture. If issues are disabled or data is unavailable, you MUST state: "Insufficient quantitative data to evaluate maintainer responsiveness." and stop.]

## 4. Repository Signal Strength
[A concise paragraph (max 100 words) synthesising stars, forks, watchers, and primary language into a signal-strength assessment. Ground every claim in the provided metadata. No external knowledge, no comparisons to repos not in the context block.]

## 5. Data Limitations
[CONDITIONAL -- MANDATORY if the DATA GAPS list in the context is non-empty.]
[If DATA GAPS is empty, write only: "No data limitations identified for this analysis."]
[If DATA GAPS is non-empty, you MUST reproduce EVERY string from the DATA GAPS list verbatim, formatted as a numbered list. Do not paraphrase, summarise, or reorder them. These strings are audit records, not editorial content.]

## 6. Investment Verdict
[A single paragraph of maximum 120 words. State a clear verdict: HIGH CONVICTION BUY / MONITOR / PASS. Justify it in one sentence per signal pillar (engineering thesis, developer momentum, maintainer responsiveness). If any pillar returned "Insufficient quantitative data", your verdict MUST be MONITOR and you MUST cite the specific insufficiency as a blocking factor.]

---
*Memo generated by autonomous pipeline. All quantitative claims are machine-verified against live GitHub GraphQL data. Qualitative synthesis by LLM -- validate independently before acting.*

ABSOLUTE RULES -- VIOLATIONS WILL INVALIDATE THE MEMO
======================================================
1. NEVER invent a metric. If a number is not in the QUANTITATIVE CONTEXT block, do not write it.
2. NEVER use hedged qualitative language to cover for missing data (e.g. "likely strong adoption", "appears to be growing"). If data is missing, cite the specific data gap.
3. NEVER add sections beyond the six defined above.
4. Data Limitations strings from DATA GAPS must be reproduced character-for-character. They are evidence, not prose.
5. The word count limits are hard limits. Do not exceed them.
""".strip()


# ─────────────────────────────────────────────
# §2  Context Serialiser
# ─────────────────────────────────────────────

def _fmt_optional(value: Optional[float | int], suffix: str = "", missing: str = "N/A") -> str:
    """Format a nullable numeric value for the context block."""
    if value is None:
        return missing
    return f"{value:,}{suffix}" if isinstance(value, int) else f"{value}{suffix}"


def build_context_block(item: EnrichedItem) -> str:
    """
    Serialise an EnrichedItem into a structured plaintext context block.

    This is the *only* information Agent 3 is permitted to use. By making it
    exhaustive and explicit, we eliminate any need for the LLM to rely on
    parametric memory -- the single largest source of metric hallucination.
    """
    ci = item.classified
    raw = ci.raw
    rat = ci.rationale

    lines: list[str] = []

    # -- Header --
    lines += [
        "=============================================================",
        "   QUANTITATIVE CONTEXT -- DO NOT MODIFY",
        "   All claims in the memo must cite a value from this block",
        "=============================================================",
        "",
    ]

    # -- Signal provenance --
    lines += [
        "-- SIGNAL PROVENANCE --",
        f"Item ID          : {raw.item_id}",
        f"Source           : {raw.source.value.upper()}",
        f"Original title   : {raw.title}",
        f"URL              : {raw.url or 'Not provided'}",
        f"Ingested at      : {raw.ingested_at.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Scout confidence : {ci.confidence:.0%}",
        "",
    ]

    # -- Scout rationale --
    lines += [
        "-- SCOUT RATIONALE (Agent 1 Output) --",
        f"Summary          : {rat.one_line_summary}",
        f"Architectural shift  : {'YES' if rat.architectural_shift else 'NO'}",
        f"Developer adoption   : {'YES' if rat.developer_adoption else 'NO'}",
        f"OSS milestone        : {'YES' if rat.oss_milestone else 'NO'}",
        f"Noise indicator      : {'YES (review verdict)' if rat.noise_signal else 'NO'}",
        "",
    ]

    # -- Per-repo metrics --
    for i, m in enumerate(item.metrics, start=1):
        p  = m.profile
        cv = m.commit_velocity
        ir = m.issue_resolution

        lines += [
            f"-- REPOSITORY {i} OF {len(item.metrics)}: {p.name_with_owner} --",
            f"  GitHub URL       : {m.github_url}",
            f"  Default branch   : {p.default_branch or 'N/A'}",
            f"  Primary language : {p.primary_language or 'N/A'}",
            f"  Stars            : {_fmt_optional(p.stars)}",
            f"  Forks            : {_fmt_optional(p.forks)}",
            f"  Watchers         : {_fmt_optional(p.watchers)}",
            f"  Archived         : {'YES -- treat metrics as frozen' if p.is_archived else 'NO'}",
            f"  Empty            : {'YES -- no data available' if p.is_empty else 'NO'}",
            "",
            "  COMMIT VELOCITY",
            f"    Last 30 days   : {_fmt_optional(cv.last_30_days, ' commits')}",
            f"    Last 60 days   : {_fmt_optional(cv.last_60_days, ' commits')}",
            f"    Last 90 days   : {_fmt_optional(cv.last_90_days, ' commits')}",
            f"    Weekly avg     : {_fmt_optional(cv.weekly_avg_30d, ' commits/week (30d)')}",
            f"    Acceleration   : {cv.acceleration}",
            "",
            "  ISSUE RESOLUTION",
            f"    Issues enabled : {'YES' if ir.has_issues_enabled else 'NO -- data unavailable'}",
            f"    Sample size    : {ir.sample_size} closed issues analysed",
            f"    Total closed   : {_fmt_optional(ir.total_closed_issues)}",
            f"    Median TTR     : {_fmt_optional(ir.median_days, ' days')}",
            f"    P25 TTR        : {_fmt_optional(round(ir.p25_hours / 24, 2) if ir.p25_hours else None, ' days')}",
            f"    P75 TTR        : {_fmt_optional(round(ir.p75_hours / 24, 2) if ir.p75_hours else None, ' days')}",
            f"    Grade          : {ir.responsiveness_grade}",
            "",
        ]

        # -- Data gaps --
        if m.data_gaps:
            lines.append("  DATA GAPS (reproduce verbatim in section 5 of the memo):")
            for idx, gap in enumerate(m.data_gaps, start=1):
                lines.append(f"    [{idx}] {gap}")
            lines.append("")
        else:
            lines.append("  DATA GAPS: NONE")
            lines.append("")

    # -- Footer --
    lines += [
        "-- END OF QUANTITATIVE CONTEXT --",
        "",
        "INSTRUCTION REMINDER:",
        "  * Every metric you cite in the memo must match a value above exactly.",
        "  * If a value is 'N/A', state 'Insufficient quantitative data'.",
        "  * Reproduce DATA GAP strings character-for-character in section 5.",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────
# §3  User Message Builder
# ─────────────────────────────────────────────

def build_user_message(item: EnrichedItem) -> str:
    """
    Compose the full user turn: a brief instruction frame followed by the
    exhaustive context block. The instruction frame is intentionally minimal --
    all behavioural rules live in the system prompt.
    """
    repo_name = (
        item.primary_metrics.profile.name_with_owner
        if item.primary_metrics
        else item.classified.raw.title[:40]
    )
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    instruction = textwrap.dedent(f"""
        Draft the investment memo for **{repo_name}** using the quantitative
        context block below. Substitute {{REPO_NAME}} with `{repo_name}`,
        {{DATE}} with `{date_str}`, and {{SOURCE}} with
        `{item.classified.raw.source.value.upper()} / {item.classified.raw.url or 'N/A'}`.

        Follow the system prompt format rules without deviation.
    """).strip()

    context = build_context_block(item)
    return f"{instruction}\n\n{context}"


# ─────────────────────────────────────────────
# §4  Core Memo Drafting Function
# ─────────────────────────────────────────────

def draft_memo(
    item:   EnrichedItem,
    client: Groq,
    *,
    verbose: bool = True,
) -> str:
    """
    Accept an EnrichedItem and return a grounded Markdown investment memo.

    Raises:
        ValueError: If `item.metrics` is empty (no quantitative grounding
                    available -- memo drafting is not permitted per hallucination guardrails).
        groq.APIError: On network or API failures (caller should retry).
    """
    # -- Guard: refuse to draft without quantitative grounding --
    if not item.metrics:
        raise ValueError(
            f"Cannot draft memo for '{item.classified.raw.title[:60]}': "
            "EnrichedItem.metrics is empty. Agent 3 requires quantitative "
            "grounding from Agent 2. Aborting per hallucination guardrails."
        )

    repo_name = item.primary_metrics.profile.name_with_owner  # type: ignore[union-attr]

    if verbose:
        all_gaps = [gap for m in item.metrics for gap in m.data_gaps]
        print(f"  -> Drafting memo for: {repo_name}")
        print(f"     Repos in scope    : {len(item.metrics)}")
        print(f"     Data gaps         : {len(all_gaps)}")
        if all_gaps:
            print(f"     WARNING: Data gaps will force MONITOR verdict if critical.")

    user_message = build_user_message(item)

    if verbose:
        print(f"     Context block     : {len(user_message):,} chars  ->  calling API...")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=2048,
        messages=[
            {"role": "system", "content": PARTNER_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    memo = response.choices[0].message.content.strip()

    if verbose:
        print(f"     Memo drafted      : {len(memo):,} chars  |  "
              f"{response.usage.completion_tokens} output tokens")

    return memo


# ─────────────────────────────────────────────
# §5  Mock EnrichedItem (standalone testing)
# ─────────────────────────────────────────────

def build_mock_enriched_item(*, include_data_gaps: bool = False) -> EnrichedItem:
    """
    Constructs a fully-populated EnrichedItem that mirrors what Agents 1 & 2
    would emit for `astral-sh/uv`, without requiring the upstream pipeline.

    Args:
        include_data_gaps: If True, injects a realistic data gap (issues
                           disabled) to validate section 5 and MONITOR verdict paths.
    """
    now = datetime.now(timezone.utc)

    # -- Agent 1 output --
    raw = RawItem(
        item_id="hn_41903999",
        source=DataSource.HACKERNEWS,
        title=(
            "uv 0.4: Rust-based pip/virtualenv replacement ships workspace "
            "support and PEP 723 -- resolves 220 deps in 38ms"
        ),
        body=(
            "Astral's uv 0.4 adds multi-project workspace support, a lock-file "
            "format, and PEP 723 inline script metadata. Resolver benchmarks: "
            "Django + 220 deps resolves in 38ms vs pip's 4.1s (108x faster). "
            "The project has 38k GitHub stars and merges 30+ PRs/week. "
            "It is rapidly replacing pip in modern Python toolchains."
        ),
        url="https://astral.sh/blog/uv-0-4",
        github_urls=["https://github.com/astral-sh/uv"],
        ingested_at=now - timedelta(hours=2),
        raw_score=4102,
    )
    classified = ClassifiedItem(
        raw=raw,
        classification=Classification.SIGNAL,
        confidence=0.97,
        rationale=ScoutRationale(
            architectural_shift=True,
            developer_adoption=True,
            oss_milestone=True,
            noise_signal=False,
            one_line_summary=(
                "uv replaces pip and virtualenv with a Rust resolver that is "
                "two orders of magnitude faster, unifying the fragmented Python "
                "packaging toolchain into a single zero-config binary."
            ),
        ),
        classified_at=now - timedelta(hours=2),
    )

    # -- Agent 2 output --
    profile = RepoProfile(
        name_with_owner="astral-sh/uv",
        default_branch="main",
        primary_language="Rust",
        stars=38_412,
        forks=1_089,
        watchers=312,
        is_archived=False,
        is_empty=False,
    )

    commit_velocity = CommitVelocity(
        last_30_days=165,
        last_60_days=298,
        last_90_days=421,
    )

    if include_data_gaps:
        issue_resolution = IssueResolutionMetrics(
            sample_size=0,
            median_hours=None,
            p25_hours=None,
            p75_hours=None,
            total_closed_issues=None,
            has_issues_enabled=False,
        )
        data_gaps = [
            "Issue resolution metrics unavailable: the repository has disabled the Issues tab.",
            "TTR statistics are low-confidence: only 0 valid issue records (threshold is 5). "
            "Median may not be representative.",
        ]
    else:
        issue_resolution = IssueResolutionMetrics(
            sample_size=50,
            median_hours=19.4,
            p25_hours=6.2,
            p75_hours=61.8,
            total_closed_issues=2_847,
            has_issues_enabled=True,
        )
        data_gaps = []

    rate_limit = RateLimitSnapshot(
        limit=5000,
        remaining=4_831,
        used=169,
        reset_at=now + timedelta(hours=1),
    )

    metrics = QuantitativeMetrics(
        github_url="https://github.com/astral-sh/uv",
        profile=profile,
        commit_velocity=commit_velocity,
        issue_resolution=issue_resolution,
        rate_limit=rate_limit,
        fetched_at=now - timedelta(minutes=5),
        data_gaps=data_gaps,
    )

    return EnrichedItem(classified=classified, metrics=[metrics])


# ─────────────────────────────────────────────
# §6  Entrypoint
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 68)
    print("  Agent 3: The Partner -- Phase 3 Test Run")
    print("=" * 68)

    # CLI flag: `python agent3_partner.py --with-gaps` to test the
    # Data Limitations + MONITOR verdict code path.
    with_gaps = "--with-gaps" in sys.argv

    mode_label = "WITH data gaps (issues disabled)" if with_gaps else "CLEAN (full data)"
    print(f"\n  Test mode: {mode_label}")
    print("  Building mock EnrichedItem...")
    mock_item = build_mock_enriched_item(include_data_gaps=with_gaps)

    # Validate early-exit guard.
    print("\n  Validating early-exit guard (empty metrics)...")
    empty_item = EnrichedItem(classified=mock_item.classified, metrics=[])
    try:
        draft_memo(empty_item, Groq(), verbose=False)  # type: ignore
        print("  FAIL: Guard should have raised ValueError.")
    except ValueError as exc:
        print(f"  PASS: Guard triggered correctly: {exc}")

    # Draft the real memo.
    print(f"\n  Initialising Groq client...")
    client = Groq()  # reads GROQ_API_KEY from env

    print()
    memo = draft_memo(mock_item, client, verbose=True)

    separator = "-" * 68
    print(f"\n{separator}")
    print("  GENERATED INVESTMENT MEMO")
    print(separator)
    print(memo)
    print(separator)

    slug = "with_gaps" if with_gaps else "clean"
    out_path = f"memo_astral_uv_{slug}.md"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(memo)
    print(f"\n  Memo saved to: {out_path}")
    print("\n  Phase 3 complete. Ready for Phase 4 end-to-end orchestration.\n")
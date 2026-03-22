from __future__ import annotations

import os
import statistics
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import MagicMock

import requests
from pydantic import BaseModel, Field, computed_field
from agent1_scout import ClassifiedItem, Classification, DataSource, RawItem, ScoutRationale



GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
COMMITS_PER_PAGE = 100
ISSUES_TO_FETCH = 50


def _github_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Github-Next-Global-ID": "1",  
    }



REPO_METRICS_QUERY = """
query RepoMetrics(
  $owner: String!
  $name:  String!
  $since: GitTimestamp!
  $issueCount: Int!
) {
  repository(owner: $owner, name: $name) {
    nameWithOwner
    isArchived
    isEmpty
    defaultBranchRef {
      name
      target {
        ... on Commit {
          history(since: $since) {
            totalCount
          }
        }
      }
    }
    commits30: defaultBranchRef {
      target {
        ... on Commit {
          history(since: $since30) {
            totalCount
          }
        }
      }
    }
    commits60: defaultBranchRef {
      target {
        ... on Commit {
          history(since: $since60) {
            totalCount
          }
        }
      }
    }
    issues(
      last: $issueCount
      states: CLOSED
      orderBy: { field: UPDATED_AT, direction: ASC }
    ) {
      totalCount
      nodes {
        createdAt
        closedAt
        title
      }
    }
    hasIssuesEnabled
    stargazerCount
    forkCount
    watchers { totalCount }
    primaryLanguage { name }
    description
    homepageUrl
    repositoryTopics(first: 10) {
      nodes {
        topic { name }
      }
    }
  }
  rateLimit {
    limit
    remaining
    resetAt
    used
  }
}
""".strip()


REPO_METRICS_QUERY_FULL = """
query RepoMetrics(
  $owner:       String!
  $name:        String!
  $since90:     GitTimestamp!
  $since60:     GitTimestamp!
  $since30:     GitTimestamp!
  $issueCount:  Int!
) {
  repository(owner: $owner, name: $name) {
    nameWithOwner
    isArchived
    isEmpty
    defaultBranchRef {
      name
    }
    commits90: defaultBranchRef {
      target {
        ... on Commit {
          history(since: $since90) { totalCount }
        }
      }
    }
    commits60: defaultBranchRef {
      target {
        ... on Commit {
          history(since: $since60) { totalCount }
        }
      }
    }
    commits30: defaultBranchRef {
      target {
        ... on Commit {
          history(since: $since30) { totalCount }
        }
      }
    }
    issues(
      last: $issueCount
      states: CLOSED
      orderBy: { field: UPDATED_AT, direction: ASC }
    ) {
      totalCount
      nodes {
        createdAt
        closedAt
      }
    }
    hasIssuesEnabled
    stargazerCount
    forkCount
    watchers { totalCount }
    primaryLanguage { name }
    description
    homepageUrl
    repositoryTopics(first: 10) {
      nodes {
        topic { name }
      }
    }
  }
  rateLimit {
    limit
    remaining
    resetAt
    used
  }
}
""".strip()


class CommitVelocity(BaseModel):
    last_30_days: Optional[int] = Field(None, description="Commits merged in the last 30 days.")
    last_60_days: Optional[int] = Field(None, description="Commits merged in the last 60 days.")
    last_90_days: Optional[int] = Field(None, description="Commits merged in the last 90 days.")

    @computed_field
    @property
    def weekly_avg_30d(self) -> Optional[float]:
        if self.last_30_days is None:
            return None
        return round(self.last_30_days / 4.29, 2)   # 30 / 7

    @computed_field
    @property
    def acceleration(self) -> Optional[str]:
    
        if self.last_30_days is None or self.last_60_days is None:
            return "INSUFFICIENT"
        pace_30 = self.last_30_days / 30
        pace_60 = (self.last_60_days - self.last_30_days) / 30   # prior 30d
        if pace_60 == 0:
            return "ACCELERATING" if pace_30 > 0 else "STEADY"
        ratio = pace_30 / pace_60
        if ratio > 1.10:
            return "ACCELERATING"
        if ratio < 0.90:
            return "DECELERATING"
        return "STEADY"


class IssueResolutionMetrics(BaseModel):

    sample_size:             int            = Field(...,  description="Number of closed issues analysed.")
    median_hours:            Optional[float]= Field(None, description="Median time-to-resolution in hours.")
    p25_hours:               Optional[float]= Field(None, description="25th-percentile resolution time (hours).")
    p75_hours:               Optional[float]= Field(None, description="75th-percentile resolution time (hours).")
    total_closed_issues:     Optional[int]  = Field(None, description="Lifetime closed issue count from GitHub.")
    has_issues_enabled:      bool           = Field(True, description="False when the repo has disabled the Issues tab.")

    @computed_field
    @property
    def median_days(self) -> Optional[float]:
        if self.median_hours is None:
            return None
        return round(self.median_hours / 24, 2)

    @computed_field
    @property
    def responsiveness_grade(self) -> str:
        if self.median_hours is None:
            return "UNKNOWN"
        d = self.median_hours / 24
        if d < 1:    return "A"
        if d < 7:    return "B"
        if d < 30:   return "C"
        return "D"


class RepoProfile(BaseModel):
    name_with_owner:  str            = Field(..., description="'owner/repo' canonical form.")
    default_branch:   Optional[str]  = Field(None)
    primary_language: Optional[str]  = Field(None)
    description:      Optional[str]  = Field(None, description="GitHub repository description (the one-liner under the repo name).")
    topics:           list[str]      = Field(default_factory=list, description="GitHub topic tags on the repository.")
    homepage:         Optional[str]  = Field(None, description="Project homepage URL if set.")
    stars:            Optional[int]  = Field(None)
    forks:            Optional[int]  = Field(None)
    watchers:         Optional[int]  = Field(None)
    is_archived:      bool           = Field(False)
    is_empty:         bool           = Field(False)


class RateLimitSnapshot(BaseModel):
    limit:      int
    remaining:  int
    used:       int
    reset_at:   datetime


class QuantitativeMetrics(BaseModel):
    github_url:        str                             = Field(..., description="The repo URL that was analysed.")
    profile:           RepoProfile                     = Field(..., description="Repository metadata.")
    commit_velocity:   CommitVelocity                  = Field(..., description="Rolling commit counts.")
    issue_resolution:  IssueResolutionMetrics          = Field(..., description="Closed-issue TTR statistics.")
    rate_limit:        Optional[RateLimitSnapshot]     = Field(None, description="API rate-limit state at fetch time.")
    fetched_at:        datetime                        = Field(
                           default_factory=lambda: datetime.now(timezone.utc)
                       )
    data_gaps:         list[str]                       = Field(
                           default_factory=list,
                           description=(
                               "Human-readable list of missing/unavailable data. "
                               "Agent 3 must quote each entry verbatim in the memo."
                           )
                       )


class EnrichedItem(BaseModel):
    
    classified:  ClassifiedItem              = Field(..., description="Full Agent 1 output, unchanged.")
    metrics:     list[QuantitativeMetrics]   = Field(
                     default_factory=list,
                     description="One QuantitativeMetrics entry per GitHub URL found in the signal."
                 )

    @property
    def has_sufficient_data(self) -> bool:
        
        if not self.metrics:
            return False
        return all(len(m.data_gaps) == 0 for m in self.metrics)

    @property
    def primary_metrics(self) -> Optional[QuantitativeMetrics]:
  
        return self.metrics[0] if self.metrics else None



def parse_github_url(url: str) -> tuple[str, str]:
    
    import re
    pattern = r"https?://github\.com/([^/]+)/([^/?#\.]+)"
    match = re.search(pattern, url)
    if not match:
        raise ValueError(f"Cannot parse GitHub owner/repo from: {url}")
    return match.group(1), match.group(2)



class RateLimitExceededError(Exception):

    def __init__(self, reset_at: datetime):
        self.reset_at = reset_at
        super().__init__(f"GitHub rate limit exceeded. Resets at {reset_at.isoformat()}")


def _execute_graphql(
    query:     str,
    variables: dict,
    token:     str,
    session:   requests.Session,
    *,
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> dict:
    """
    Execute a GitHub GraphQL query with exponential back-off on transient errors.

    Handles:
      - 401 Unauthorized  → propagates immediately (bad token, no point retrying)
      - 403 / 429         → raises RateLimitExceededError with reset timestamp
      - 5xx               → retries with exponential back-off
      - GraphQL `errors`  → raises ValueError with the first error message

    Returns the `data` sub-dict from the GraphQL response.
    """
    payload = {"query": query, "variables": variables}
    headers = _github_headers(token)

    for attempt in range(max_retries):
        try:
            response = session.post(
                GITHUB_GRAPHQL_URL,
                json=payload,
                headers=headers,
                timeout=30,
            )
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(backoff_base ** attempt)
                continue
            raise

       
        if response.status_code == 401:
            raise PermissionError("GitHub token is invalid or expired (HTTP 401).")

      
        if response.status_code in (403, 429):
            reset_str = response.headers.get("X-RateLimit-Reset")
            if reset_str:
                reset_dt = datetime.fromtimestamp(int(reset_str), tz=timezone.utc)
            else:
                reset_dt = datetime.now(timezone.utc) + timedelta(minutes=60)
            raise RateLimitExceededError(reset_dt)

        # ── Transient server error — back off and retry ───────────────────
        if response.status_code >= 500:
            if attempt < max_retries - 1:
                wait = backoff_base ** attempt
                print(f"    [GitHub] HTTP {response.status_code} — retrying in {wait}s…")
                time.sleep(wait)
                continue
            response.raise_for_status()

        response.raise_for_status()
        body = response.json()

       
        if "errors" in body:
            first_error = body["errors"][0]
            error_type  = first_error.get("type", "UNKNOWN")
            error_msg   = first_error.get("message", str(first_error))

        
            if error_type == "NOT_FOUND":
                raise ValueError(f"Repository not found or private: {error_msg}")
            raise ValueError(f"GitHub GraphQL error [{error_type}]: {error_msg}")

        return body.get("data", {})

    raise RuntimeError("GraphQL request failed after all retries.") 



def _calc_issue_resolution(issue_nodes: list[dict]) -> tuple[list[float], list[str]]:
    
    resolution_hours: list[float] = []
    warnings: list[str] = []

    for node in issue_nodes:
        created_raw = node.get("createdAt")
        closed_raw  = node.get("closedAt")

        if not created_raw or not closed_raw:
            warnings.append(f"Issue missing timestamp — skipped from TTR calculation.")
            continue

        try:
            created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
            closed  = datetime.fromisoformat(closed_raw.replace("Z",  "+00:00"))
        except ValueError as exc:
            warnings.append(f"Unparseable issue timestamp ({exc}) — skipped.")
            continue

        delta_hours = (closed - created).total_seconds() / 3600
        if delta_hours < 0:
            warnings.append("Issue has closedAt before createdAt — skipped (data anomaly).")
            continue

        resolution_hours.append(delta_hours)

    return resolution_hours, warnings


def _percentile(data: list[float], pct: float) -> float:
 
    if not data:
        raise ValueError("Empty list")
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = (pct / 100) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])



def analyse_repository(
    github_url: str,
    token:      str,
    session:    requests.Session,
) -> QuantitativeMetrics:
    """
    Execute the GitHub GraphQL query for a single repo URL and return
    a fully populated QuantitativeMetrics object.

    All failures are captured in `data_gaps`; the function never raises
    on missing data — it degrades gracefully and documents what was lost.
    """
    data_gaps: list[str] = []

    try:
        owner, repo_name = parse_github_url(github_url)
    except ValueError as exc:
        return QuantitativeMetrics(
            github_url=github_url,
            profile=RepoProfile(name_with_owner=github_url),
            commit_velocity=CommitVelocity(),
            issue_resolution=IssueResolutionMetrics(sample_size=0, has_issues_enabled=False),
            data_gaps=[f"CRITICAL: Could not parse GitHub URL — {exc}"],
        )

    now     = datetime.now(timezone.utc)
    since90 = (now - timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%SZ")
    since60 = (now - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
    since30 = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

    variables = {
        "owner":      owner,
        "name":       repo_name,
        "since90":    since90,
        "since60":    since60,
        "since30":    since30,
        "issueCount": ISSUES_TO_FETCH,
    }

    try:
        gql_data = _execute_graphql(
            REPO_METRICS_QUERY_FULL, variables, token, session
        )
    except RateLimitExceededError as exc:
        return QuantitativeMetrics(
            github_url=github_url,
            profile=RepoProfile(name_with_owner=f"{owner}/{repo_name}"),
            commit_velocity=CommitVelocity(),
            issue_resolution=IssueResolutionMetrics(sample_size=0, has_issues_enabled=True),
            data_gaps=[
                f"CRITICAL: GitHub rate limit exceeded. All metrics unavailable. "
                f"Quota resets at {exc.reset_at.isoformat()}."
            ],
        )
    except (ValueError, PermissionError) as exc:
        return QuantitativeMetrics(
            github_url=github_url,
            profile=RepoProfile(name_with_owner=f"{owner}/{repo_name}"),
            commit_velocity=CommitVelocity(),
            issue_resolution=IssueResolutionMetrics(sample_size=0, has_issues_enabled=True),
            data_gaps=[f"CRITICAL: GraphQL query failed — {exc}"],
        )

    repo = gql_data.get("repository", {})

 
    if repo.get("isEmpty"):
        data_gaps.append("Repository is empty — no commits or issues available.")
    if repo.get("isArchived"):
        data_gaps.append("Repository is archived — metrics reflect a frozen codebase.")

    raw_topics = repo.get("repositoryTopics", {}) or {}
    topic_names = [
        node["topic"]["name"]
        for node in raw_topics.get("nodes", [])
        if node and node.get("topic")
    ]

    profile = RepoProfile(
        name_with_owner  = repo.get("nameWithOwner", f"{owner}/{repo_name}"),
        default_branch   = (repo.get("defaultBranchRef") or {}).get("name"),
        primary_language = (repo.get("primaryLanguage") or {}).get("name"),
        description      = repo.get("description") or None,
        topics           = topic_names,
        homepage         = repo.get("homepageUrl") or None,
        stars            = repo.get("stargazerCount"),
        forks            = repo.get("forkCount"),
        watchers         = (repo.get("watchers") or {}).get("totalCount"),
        is_archived      = repo.get("isArchived", False),
        is_empty         = repo.get("isEmpty", False),
    )

    def _extract_commit_count(alias: str) -> Optional[int]:
        branch_ref = repo.get(alias)
        if not branch_ref:
            return None
        target = branch_ref.get("target", {})
        history = target.get("history", {})
        return history.get("totalCount")

    c90 = _extract_commit_count("commits90")
    c60 = _extract_commit_count("commits60")
    c30 = _extract_commit_count("commits30")

    if c90 is None:
        data_gaps.append(
            "Commit velocity unavailable: defaultBranchRef returned no data "
            "(the repo may have no default branch or commits are inaccessible)."
        )

    commit_velocity = CommitVelocity(
        last_30_days=c30,
        last_60_days=c60,
        last_90_days=c90,
    )

    has_issues = repo.get("hasIssuesEnabled", True)
    issues_obj = repo.get("issues", {})
    issue_nodes = issues_obj.get("nodes", []) if issues_obj else []
    total_closed = issues_obj.get("totalCount") if issues_obj else None

    if not has_issues:
        data_gaps.append(
            "Issue resolution metrics unavailable: the repository has disabled the Issues tab."
        )
        issue_resolution = IssueResolutionMetrics(
            sample_size=0,
            total_closed_issues=None,
            has_issues_enabled=False,
        )
    elif not issue_nodes:
        data_gaps.append(
            "Issue resolution metrics unavailable: no closed issues found in the repository."
        )
        issue_resolution = IssueResolutionMetrics(
            sample_size=0,
            total_closed_issues=total_closed or 0,
            has_issues_enabled=True,
        )
    else:
        resolution_hours, ttr_warnings = _calc_issue_resolution(issue_nodes)
        data_gaps.extend(ttr_warnings)

        if len(resolution_hours) < 5:
            data_gaps.append(
                f"TTR statistics are low-confidence: only {len(resolution_hours)} valid "
                f"issue records (threshold is 5). Median may not be representative."
            )

        if resolution_hours:
            median  = statistics.median(resolution_hours)
            p25     = _percentile(resolution_hours, 25)
            p75     = _percentile(resolution_hours, 75)
        else:
            median = p25 = p75 = None
            data_gaps.append(
                "TTR could not be computed: all issue records were invalid or malformed."
            )

        issue_resolution = IssueResolutionMetrics(
            sample_size=len(resolution_hours),
            median_hours=round(median, 2) if median is not None else None,
            p25_hours=round(p25, 2) if p25 is not None else None,
            p75_hours=round(p75, 2) if p75 is not None else None,
            total_closed_issues=total_closed,
            has_issues_enabled=True,
        )

    rate_limit_raw = gql_data.get("rateLimit")
    rate_limit_snapshot = None
    if rate_limit_raw:
        try:
            rate_limit_snapshot = RateLimitSnapshot(
                limit=rate_limit_raw["limit"],
                remaining=rate_limit_raw["remaining"],
                used=rate_limit_raw["used"],
                reset_at=datetime.fromisoformat(
                    rate_limit_raw["resetAt"].replace("Z", "+00:00")
                ),
            )
   
            pct_used = rate_limit_raw["used"] / rate_limit_raw["limit"]
            if pct_used > 0.80:
                print(
                    f"   Rate limit warning: {rate_limit_raw['remaining']} points "
                    f"remaining (resets {rate_limit_raw['resetAt']})"
                )
        except (KeyError, ValueError):
            pass  

    return QuantitativeMetrics(
        github_url=github_url,
        profile=profile,
        commit_velocity=commit_velocity,
        issue_resolution=issue_resolution,
        rate_limit=rate_limit_snapshot,
        data_gaps=data_gaps,
    )



def enrich_signal(
    item:    ClassifiedItem,
    token:   str,
    session: Optional[requests.Session] = None,
) -> EnrichedItem:
    """
    Top-level function: accepts a ClassifiedItem and returns an EnrichedItem.

    Skips analysis silently if the item is NOISE or has no GitHub targets,
    returning an EnrichedItem with an empty `metrics` list.
    """
    if not item.is_signal or not item.has_github_targets:
        return EnrichedItem(classified=item)

    own_session = session is None
    if own_session:
        session = requests.Session()

    metrics_list: list[QuantitativeMetrics] = []
    for url in item.raw.github_urls:
        print(f"  → Analysing {url}")
        metrics = analyse_repository(url, token, session)
        metrics_list.append(metrics)

        
        if metrics.data_gaps:
            for gap in metrics.data_gaps:
                prefix = "   " if not gap.startswith("CRITICAL") else "   "
                print(f"{prefix}{gap}")

    if own_session:
        session.close()

    return EnrichedItem(classified=item, metrics=metrics_list)


def run_quant_batch(
    signals: list[ClassifiedItem],
    token:   str,
    *,
    verbose: bool = True,
) -> list[EnrichedItem]:
    """Process a batch of signals from Agent 1 and return EnrichedItems."""
    enriched: list[EnrichedItem] = []
    session = requests.Session()

    for item in signals:
        if verbose:
            print(f"\n{'─'*60}")
            print(f"  [Agent 2] Enriching: {item.raw.title[:65]}")
        enriched.append(enrich_signal(item, token, session))

    session.close()
    return enriched



def _build_mock_graphql_response(
    owner: str,
    repo:  str,
    *,
    disable_issues: bool = False,
    empty_repo:     bool = False,
) -> dict:
    """
    Returns a realistic mock GraphQL `data` payload that mirrors the exact
    shape returned by REPO_METRICS_QUERY_FULL. Used when GITHUB_TOKEN is not set.
    """
    now = datetime.now(timezone.utc)


    issue_nodes = []
    for i in range(ISSUES_TO_FETCH):
        created = now - timedelta(days=90 - i)
        resolution_hours = 2 + (i % 15) * 8
        closed = created + timedelta(hours=resolution_hours)
        issue_nodes.append({
            "createdAt": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "closedAt":  closed.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

    return {
        "repository": {
            "nameWithOwner":    f"{owner}/{repo}",
            "isArchived":       False,
            "isEmpty":          empty_repo,
            "defaultBranchRef": {"name": "main"},
            "commits90": {
                "target": {"history": {"totalCount": 420}}
            },
            "commits60": {
                "target": {"history": {"totalCount": 280}}
            },
            "commits30": {
                "target": {"history": {"totalCount": 165}}
            },
            "issues": None if disable_issues else {
                "totalCount": 812,
                "nodes": issue_nodes,
            },
            "hasIssuesEnabled": not disable_issues,
            "stargazerCount":   38_400,
            "forkCount":        1_204,
            "watchers":         {"totalCount": 892},
            "primaryLanguage":  {"name": "Rust"},
        },
        "rateLimit": {
            "limit":     5000,
            "remaining": 4873,
            "used":      127,
            "resetAt":   (now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }


def _patch_session_with_mock(
    session:        requests.Session,
    owner:          str,
    repo:           str,
    disable_issues: bool = False,
) -> None:
    
    mock_data = _build_mock_graphql_response(owner, repo, disable_issues=disable_issues)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": mock_data}
    mock_response.raise_for_status = lambda: None

    session.post = MagicMock(return_value=mock_response)



def print_enriched_item(ei: EnrichedItem) -> None:
    """Human-readable summary of an EnrichedItem for Phase 2 validation."""
    title = ei.classified.raw.title
    print(f"\n{'═'*65}")
    print(f"  ENRICHED SIGNAL: {title[:60]}")
    print(f"{'═'*65}")

    if not ei.metrics:
        print("    No metrics — item was NOISE or had no GitHub URLs.")
        return

    for m in ei.metrics:
        p = m.profile
        cv = m.commit_velocity
        ir = m.issue_resolution

        print(f"\n  Repo         : {p.name_with_owner}  [{p.primary_language or 'unknown lang'}]")
        print(f"  Default branch: {p.default_branch or 'N/A'}  |  Archived: {p.is_archived}")
        print(f"  Stars / Forks : {p.stars:,}   /  {p.forks:,} 🍴" if p.stars else "  Stars / Forks : N/A")

        print(f"\n  ── Commit Velocity ──────────────────────────────")
        print(f"  Last 30 days  : {cv.last_30_days}")
        print(f"  Last 60 days  : {cv.last_60_days}")
        print(f"  Last 90 days  : {cv.last_90_days}")
        print(f"  Weekly avg    : {cv.weekly_avg_30d} commits/week (30d window)")
        print(f"  Momentum      : {cv.acceleration}")

        print(f"\n  ── Issue Resolution ─────────────────────────────")
        print(f"  Issues enabled: {ir.has_issues_enabled}")
        print(f"  Sample size   : {ir.sample_size} / {ISSUES_TO_FETCH} requested")
        print(f"  Total closed  : {ir.total_closed_issues}")
        print(f"  Median TTR    : {ir.median_days} days  (grade: {ir.responsiveness_grade})")
        print(f"  IQR           : P25={round(ir.p25_hours/24, 2) if ir.p25_hours else 'N/A'}d  –  P75={round(ir.p75_hours/24, 2) if ir.p75_hours else 'N/A'}d")

        if m.rate_limit:
            rl = m.rate_limit
            print(f"\n  ── Rate Limit ───────────────────────────────────")
            print(f"  {rl.remaining}/{rl.limit} points remaining  (resets {rl.reset_at.strftime('%H:%M UTC')})")

        if m.data_gaps:
            print(f"\n  ── Data Gaps (Agent 3 must surface these) ───────")
            for gap in m.data_gaps:
                print(f"   {gap}")



if __name__ == "__main__":
    print("=" * 65)
    print("  Agent 2: The Quantitative Engineer — Phase 2 Test Run")
    print("=" * 65)

 

    def _make_signal(item_id, title, github_urls) -> ClassifiedItem:
        raw = RawItem(
            item_id=item_id,
            source=DataSource.HACKERNEWS,
            title=title,
            github_urls=github_urls,
        )
        return ClassifiedItem(
            raw=raw,
            classification=Classification.SIGNAL,
            confidence=0.95,
            rationale=ScoutRationale(
                architectural_shift=True,
                developer_adoption=True,
                oss_milestone=True,
                noise_signal=False,
                one_line_summary="Mock signal for Phase 2 testing.",
            ),
        )

    mock_signals = [
        _make_signal(
            "mock_001",
            "uv 0.4: Rust-based pip replacement, 100× faster resolver",
            ["https://github.com/astral-sh/uv"],
        ),
        _make_signal(
            "mock_002",
            "Deno 2.0: full npm compat, 2.1× faster cold starts",
            ["https://github.com/denoland/deno"],
        ),
    ]
    mock_noise_item = ClassifiedItem(
        raw=RawItem(
            item_id="mock_noise",
            source=DataSource.HACKERNEWS,
            title="Acme AI raises $200M Series C",
            github_urls=[],
        ),
        classification=Classification.NOISE,
        confidence=0.99,
        rationale=ScoutRationale(
            architectural_shift=False, developer_adoption=False,
            oss_milestone=False, noise_signal=True,
            one_line_summary="Funding announcement with no engineering substance.",
        ),
    )

    token = os.environ.get("GITHUB_TOKEN", "")
    use_mock = not token

    if use_mock:
        print("\n  GITHUB_TOKEN not set — running with mock GraphQL responses.\n")

    session = requests.Session()
    enriched_results: list[EnrichedItem] = []

    for idx, signal in enumerate(mock_signals):
        print(f"\n{'─'*65}")
        print(f"  [Agent 2] Enriching item {idx + 1}/{len(mock_signals)}: {signal.raw.title[:60]}")

        if use_mock:
            owner, repo = parse_github_url(signal.raw.github_urls[0])
            _patch_session_with_mock(session, owner, repo, disable_issues=(idx == 1))

        ei = enrich_signal(signal, token or "MOCK", session)
        enriched_results.append(ei)
        print_enriched_item(ei)

    print(f"\n{'─'*65}")
    print("  [Agent 2] Processing NOISE item (should be skipped)…")
    noise_result = enrich_signal(mock_noise_item, token or "MOCK", session)
    print(f"  Metrics list length: {len(noise_result.metrics)}  (expected: 0) ")

    session.close()

    print(f"\n{'═'*65}")
    print(f"  RESULTS: {len(enriched_results)} EnrichedItem(s) ready for Agent 3.")
    print("  Phase 2 complete. EnrichedItems ready for Agent 3 (The Partner).")
    print(f"{'═'*65}\n")
"""
CineMatch - Production Recommendation Engine
============================================
Syllabus Reference: Units 4 & 5 — Model Inference, Metric Evaluation,
                    Database Caching, Weighted Score Aggregation

This module implements a two-path recommendation strategy:

  Path A — Aggregated cosine similarity lookup from CachedRecommendation
            (PostgreSQL) using multi-source weighted ranking across the
            user's entire recent watchlist window (last 10 items).

  Path B — Live TMDB API fallback that activates when the DB cache table
            has insufficient coverage (new releases, empty watchlist, etc.)

Architecture:
  get_recommendation_ids()  →  returns list[int] of target media IDs
  _path_b_similar()         →  TMDB /similar endpoint fallback
  _path_b_trending()        →  TMDB /trending endpoint cold-start fallback
"""

import os
import requests
from django.db.models import (
    Sum, Avg, Count, FloatField, ExpressionWrapper
)

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "41fc74ce5602882786e1e9d4933fdcc6")
TMDB_BASE    = "https://api.themoviedb.org/3"
REQUEST_TIMEOUT = 5   # seconds — hard cap to protect Gunicorn worker threads


def get_recommendation_ids(user, media_type="movie", limit=8):
    """
    Core recommendation resolver. Returns an ordered list of target media IDs.

    Path A  — Multi-source weighted aggregation from CachedRecommendation:
              For every candidate target_id, SUM all cosine similarity scores
              across every watchlist source that recommends it, then apply a
              multi-source bonus multiplier. This naturally surfaces items
              that are similar to several things the user already loves.

    Path B  — TMDB API fallback for unindexed titles or empty watchlists.

    Args:
        user:        Django User instance (request.user)
        media_type:  "movie" | "tv"
        limit:       Maximum number of IDs to return (default 8)

    Returns:
        list[int] of recommended media IDs, ordered by relevance score.
    """
    from core.models import CachedRecommendation, MovieWatchlist, WatchedHistory

    # ── STEP 1: Recent watchlist signal window (last 10 items) ──────────────
    recent_ids = list(
        MovieWatchlist.objects
        .filter(user=user, media_type=media_type)
        .order_by("-added_at")
        .values_list("media_id", flat=True)[:10]
    )

    if not recent_ids:
        # Cold-start: no watchlist at all → trending fallback
        return _path_b_trending(media_type, limit)

    # ── STEP 2: Build exclusion set (already bookmarked + watched) ──────────
    watched_ids = set(
        WatchedHistory.objects
        .filter(user=user, media_type=media_type)
        .values_list("media_id", flat=True)
    )
    excluded_ids = set(recent_ids) | watched_ids

    # ── STEP 3: Multi-source weighted aggregation (Path A) ──────────────────
    #
    # SQL equivalent (PostgreSQL):
    #
    #   SELECT
    #       target_id,
    #       SUM(score)                              AS total_score,
    #       COUNT(source_id)                        AS source_count,
    #       SUM(score) * (1.0 + COUNT(*) * 0.1)    AS weighted_score
    #   FROM core_cachedrecommendation
    #   WHERE source_id IN (:recent_ids)
    #     AND media_type = :media_type
    #     AND target_id  NOT IN (:excluded_ids)
    #   GROUP BY target_id
    #   ORDER BY weighted_score DESC
    #   LIMIT :limit;
    #
    # Design rationale:
    #   •  SUM(score) rewards items that appear across multiple source items.
    #   •  The (1 + count * 0.1) multiplier gives a 10% bonus per additional
    #      corroborating watchlist source without overwhelming single-source
    #      high-similarity matches (tunable constant).
    #   •  NOT IN (excluded_ids) is evaluated in Python before hitting the DB,
    #      avoiding a correlated subquery on potentially large watched tables.

    qs = (
        CachedRecommendation.objects
        .filter(
            source_id__in=recent_ids,
            media_type=media_type,
        )
        .exclude(target_id__in=excluded_ids)
        .values("target_id")
        .annotate(
            total_score=Sum("score"),
            source_count=Count("source_id"),
            avg_score=Avg("score"),
            weighted_score=ExpressionWrapper(
                Sum("score") * (1.0 + Count("source_id") * 0.1),
                output_field=FloatField()
            )
        )
        .order_by("-weighted_score")
        [:limit]
    )

    result_ids = [row["target_id"] for row in qs]

    # ── STEP 4: Hard fallback — cache table has no matches at all ───────────
    if not result_ids:
        return _path_b_similar(recent_ids[0], media_type, limit, exclude=excluded_ids)

    # ── STEP 5: Soft top-up — fewer results than requested ──────────────────
    if len(result_ids) < limit:
        shortfall  = limit - len(result_ids)
        already    = set(result_ids) | excluded_ids
        api_top_up = _path_b_similar(
            recent_ids[0], media_type, shortfall, exclude=already
        )
        result_ids += api_top_up

    return result_ids


# ─────────────────────────────────────────────────────────────────────────────
# PATH B — TMDB API Fallback Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _path_b_similar(seed_id, media_type, limit, exclude=None):
    """
    Fetches similar titles from TMDB's /similar endpoint.
    Activated when CachedRecommendation has insufficient coverage for a seed.

    Args:
        seed_id:    TMDB ID of the most recently added watchlist item.
        media_type: "movie" | "tv"
        limit:      Maximum number of IDs to return.
        exclude:    Set of IDs to omit from results (already seen/saved).

    Returns:
        list[int] of TMDB media IDs.
    """
    exclude   = exclude or set()
    endpoint  = "movie" if media_type == "movie" else "tv"
    url       = f"{TMDB_BASE}/{endpoint}/{seed_id}/similar"
    try:
        resp = requests.get(
            url,
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [r["id"] for r in results if r["id"] not in exclude][:limit]
    except Exception as e:
        print(f"[Path B] TMDB /similar failed for {seed_id}: {e}")
        return []


def _path_b_trending(media_type, limit):
    """
    Empty-watchlist cold-start fallback. Returns TMDB weekly trending titles
    so that new users always see a populated For You feed.

    Args:
        media_type: "movie" | "tv"
        limit:      Maximum number of IDs to return.

    Returns:
        list[int] of TMDB media IDs.
    """
    endpoint = "movie" if media_type == "movie" else "tv"
    url      = f"{TMDB_BASE}/trending/{endpoint}/week"
    try:
        resp = requests.get(
            url,
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        return [r["id"] for r in resp.json().get("results", [])][:limit]
    except Exception as e:
        print(f"[Path B] TMDB /trending failed: {e}")
        return []

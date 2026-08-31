"""
CineMatch Project - Phase 2 & 4: Backend Controllers, ML Inference, and Dashboard
Syllabus Reference:
- Unit 9.1 & 9.2: User Accounts, Authentication, Watchlist CRUD, and Pagination
- Units 4 & 5: Model Inference, Similarity Metrics, and Vector Aggregation
- Unit 7: REST API Ingestion pipelines, dynamic URL construction and encoding
"""

import os
import pickle
import requests
import urllib.parse
import numpy as np
import pandas as pd
import joblib
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login
from core.forms import CineMatchRegistrationForm, CineMatchLoginForm
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_protect
from django.http import JsonResponse
from django.conf import settings
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.utils.html import escape

from .models import (
    UserProfile, MovieWatchlist, MediaReview, Review, WatchedHistory,
    UserReview, RecommendationFeedback,
)
from .tmdb_api import TMDBClient
from core.utils import TMDB_GENRE_MAP, get_resilient_session

# ======================================================================
# Global Memory Cache System (Unit 9.2 & 7 Optimization)
# ======================================================================
# PERSISTENT RAM CACHE: Persists fetched poster URLs across requests.
POSTER_CACHE = {}

# ======================================================================
# Sentiment Analysis Engine (ML Inference: Logistic Regression)
# ======================================================================
import re
MODEL_DIR = os.path.join(settings.BASE_DIR, 'models')
SENTIMENT_MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')

_sentiment_model = None
_vectorizer = None

def load_sentiment_assets():
    global _sentiment_model, _vectorizer
    if _sentiment_model is None or _vectorizer is None:
        try:
            if os.path.exists(SENTIMENT_MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
                _sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)
                _vectorizer = joblib.load(VECTORIZER_PATH)
                print("[SENTIMENT] Model and Vectorizer loaded successfully.")
            else:
                print("[SENTIMENT WARNING] Model files not found at:", SENTIMENT_MODEL_PATH)
        except Exception as e:
            print(f"[SENTIMENT ERROR] Failed to load sentiment assets: {e}")

def predict_sentiment(review_text):
    if not review_text or not isinstance(review_text, str):
        return "neutral", 50
    
    load_sentiment_assets()
    if _sentiment_model is None or _vectorizer is None:
        return "neutral", 50
        
    try:
        # Preprocess text (clean HTML and non-alphabetic chars)
        cleaned = re.sub(r'<[^>]*>', ' ', review_text)
        cleaned = cleaned.lower()
        cleaned = re.sub(r'[^a-zA-Z\s]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if not cleaned:
            return "neutral", 50
            
        # Transform and Predict
        vec = _vectorizer.transform([cleaned])
        prediction = _sentiment_model.predict(vec)[0]
        
        try:
            proba = _sentiment_model.predict_proba(vec)[0]
            score = int(proba[1] * 100)
        except Exception:
            score = 100 if prediction == 1 else 0
            
        label = "positive" if prediction == 1 else "negative"
        return label, score
    except Exception as e:
        print(f"[SENTIMENT ERROR] Prediction failed: {e}")
        return "neutral", 50

# Global variables to cache high-dimensional vector similarity matrices
MOVIE_DICT = None
MOVIE_SIMILARITY = None
TV_DICT = None
TV_SIMILARITY = None

def get_cached_poster(client, media_id, media_type):
    """
    Syllabus Reference: Unit 9.2 (Queryset Optimization / Caching)
    Checks if a unique combined string key exists in POSTER_CACHE.
    If yes, returns the value immediately (0ms lookup).
    If no, executes client.get_media_assets() with a strict timeout parameter
    to protect the active thread pool from starvation, and stores the result in RAM.
    """
    cache_key = f"{media_type}_{media_id}"
    if cache_key in POSTER_CACHE:
        return POSTER_CACHE[cache_key]
    
    # Pre-saved local Unsplash fallback cover links if API is offline or times out
    fallback_urls = {
        'movie': 'https://images.unsplash.com/photo-1542204172-e7052809f852?q=80&w=400&auto=format&fit=crop',
        'tv': 'https://images.unsplash.com/photo-1593305841991-05c297ba4575?q=80&w=400&auto=format&fit=crop'
    }
    
    try:
        # Enforce strict timeout block to avoid blocking Django server threads
        poster_url = client.get_media_assets(media_id, media_type, timeout=3.0)
        if poster_url:
            POSTER_CACHE[cache_key] = poster_url
            return poster_url
    except Exception as e:
        print(f"[CACHE ENGINE] Exception or timeout for key {cache_key}: {e}")
        
    return fallback_urls.get(media_type)


def load_ml_models():
    """
    Syllabus Reference: Units 4 & 5 Model Loading
    Caches the pre-computed Bag of Words similarity arrays and metadata
    dictionaries into process memory to eliminate per-request disk I/O.
    """
    global MOVIE_DICT, MOVIE_SIMILARITY, TV_DICT, TV_SIMILARITY
    if MOVIE_DICT is None:
        try:
            movie_dict_path = os.path.join(MODEL_DIR, 'movie_dict.pkl')
            movie_sim_path = os.path.join(MODEL_DIR, 'similarity.pkl')
            tv_dict_path = os.path.join(MODEL_DIR, 'tv_dict.pkl')
            tv_sim_path = os.path.join(MODEL_DIR, 'tv_similarity.pkl')

            with open(movie_dict_path, 'rb') as f:
                MOVIE_DICT = pickle.load(f)
            with open(movie_sim_path, 'rb') as f:
                MOVIE_SIMILARITY = pickle.load(f)
            with open(tv_dict_path, 'rb') as f:
                TV_DICT = pickle.load(f)
            with open(tv_sim_path, 'rb') as f:
                TV_SIMILARITY = pickle.load(f)

            print("[ML ENGINE] All vector pickles cached into memory successfully.")
        except Exception as e:
            print("[ML ENGINE] Error loading pre-computed similarity pickles:", e)

def get_genre_fallback_recommendations(recent_id, df, id_col, watchlist_indices, media_type, client):
    recent_genres = []
    from core.models import CachedMedia
    try:
        cached_item = CachedMedia.objects.filter(media_id=recent_id, media_type=media_type).first()
        if cached_item and cached_item.data:
            recent_genres = [g.get('name', '').lower() for g in cached_item.data.get('genres', [])]
    except Exception:
        pass
        
    if not recent_genres:
        match_row = df[df[id_col] == recent_id]
        if not match_row.empty:
            tags_val = str(match_row.iloc[0]['tags']).lower()
            for g in ['action', 'adventure', 'fantasy', 'sciencefiction', 'crime', 'drama', 'comedy', 'thriller', 'romance', 'animation', 'family', 'mystery']:
                if g in tags_val:
                    recent_genres.append(g)
                    
    genre_matched_indices = []
    if recent_genres:
        for idx, row in df.iterrows():
            if idx not in watchlist_indices:
                tags_val = str(row['tags']).lower()
                if any(g[:4] in tags_val for g in recent_genres):
                    genre_matched_indices.append(idx)
                    if len(genre_matched_indices) >= 8:
                        break
                        
    if len(genre_matched_indices) < 8:
        for idx in range(len(df)):
            if idx not in watchlist_indices and idx not in genre_matched_indices:
                genre_matched_indices.append(idx)
                if len(genre_matched_indices) >= 8:
                    break
                    
    recommendations = df.iloc[genre_matched_indices].to_dict(orient='records')
    for rec in recommendations:
        media_id = rec[id_col]
        title_text = rec.get('title') or rec.get('name') or 'Unknown Title'
        rec['title'] = title_text
        rec['poster_url'] = get_cached_poster(client, media_id, media_type)
        rec['watch_link'] = client.get_streaming_or_theatre_links(title_text, media_type, False)
        
    return recommendations


def get_recommendations(user_watchlist_ids, media_type='movie', user=None):
    """
    Main Vector Aggregation & Content-Based Recommendation Engine.
    """
    client = TMDBClient()

    if media_type == 'movie':
        data_dict = MOVIE_DICT
        sim_matrix = MOVIE_SIMILARITY
        id_col = 'movie_id'
    else:
        data_dict = TV_DICT
        sim_matrix = TV_SIMILARITY
        id_col = 'id'

    if data_dict is None or sim_matrix is None:
        # Falls back to database-cached pre-computed recommendations or TMDB similar if pickles are excluded
        try:
            from django.db.models import Sum, Count
            from core.models import CachedRecommendation, CachedMedia
            
            excluded_ids = set(user_watchlist_ids)
            
            qs = (
                CachedRecommendation.objects
                .filter(source_id__in=user_watchlist_ids[:10], media_type=media_type)
                .exclude(target_id__in=excluded_ids)
                .values("target_id")
                .annotate(total_score=Sum("score"))
                .order_by("-total_score")
                [:8]
            )
            
            if qs.exists():
                recommendations = []
                for row in qs:
                    tid = row["target_id"]
                    cached = CachedMedia.objects.filter(media_id=tid, media_type=media_type).first()
                    title_text = cached.data.get('title') or cached.data.get('name') if (cached and cached.data) else 'Unknown Title'
                    recommendations.append({
                        id_col: tid,
                        'title': title_text,
                        'poster_url': get_cached_poster(client, tid, media_type),
                        'watch_link': client.get_streaming_or_theatre_links(title_text, media_type, False)
                    })
                return recommendations
        except Exception as dbe:
            print(f"[DB FALLBACK ERROR] Database CachedRecommendation fetch failed: {dbe}")

        # Fallback to similar/popular items dynamically if no database recommendations exist
        if user_watchlist_ids:
            recent_id = user_watchlist_ids[0]
            if media_type == 'movie':
                results = client.get_similar_movies(recent_id)[:8]
                for r in results:
                    r['movie_id'] = r.get('id', r.get('movie_id'))
                    r['title'] = r.get('title', 'Unknown Title')
                    r['poster_url'] = get_cached_poster(client, r['movie_id'], 'movie')
                    r['watch_link'] = client.get_streaming_or_theatre_links(r['title'], 'movie', False)
                return results
            else:
                similar_url = f"{client.base_url}/tv/{recent_id}/similar"
                params = {
                    'api_key': settings.TMDB_API_KEY,
                    'language': 'en-US',
                    'page': 1
                }
                results = []
                try:
                    resp = get_resilient_session().get(similar_url, params=params, timeout=5.0)
                    if resp.status_code == 200:
                        raw_results = resp.json().get('results', [])
                        for item in raw_results[:8]:
                            show_id = item.get('id')
                            title = item.get('name', 'Unknown Title')
                            results.append({
                                'id': show_id,
                                'title': title,
                                'poster_url': get_cached_poster(client, show_id, 'tv'),
                                'watch_link': client.get_streaming_or_theatre_links(title, 'tv', False)
                            })
                except Exception as ex:
                    print(f"[TV FALLBACK ERROR] Failed to fetch similar TV shows: {ex}")
                
                if not results:
                    from core.utils import fetch_tmdb_catalog
                    catalog = fetch_tmdb_catalog(endpoint_type="tv", list_type="popular", page=1)
                    raw_results = catalog.get('results', [])[:8]
                    for r in raw_results:
                        show_id = r.get('id')
                        title = r.get('name', 'Unknown Title')
                        results.append({
                            'id': show_id,
                            'title': title,
                            'poster_url': get_cached_poster(client, show_id, 'tv'),
                            'watch_link': client.get_streaming_or_theatre_links(title, 'tv', False)
                        })
                return results
        else:
            # Empty watchlist cold start popular fallback
            try:
                from core.utils import fetch_tmdb_catalog
                catalog = fetch_tmdb_catalog(endpoint_type=media_type, list_type="popular", page=1)
                results = catalog.get('results', [])[:8]
                defaults = []
                for r in results:
                    title_text = r.get('title') or r.get('name') or 'Unknown Title'
                    media_id = r.get('id')
                    defaults.append({
                        id_col: media_id,
                        'title': title_text,
                        'poster_url': get_cached_poster(client, media_id, media_type),
                        'watch_link': client.get_streaming_or_theatre_links(title_text, media_type, False),
                    })
                return defaults
            except Exception:
                return []

    df = pd.DataFrame(data_dict)

    # 1. COLD START LAYER 1: Empty Watchlist -> Global Popularity
    if not user_watchlist_ids:
        trending_df = df.sort_values(by='popularity', ascending=False) if 'popularity' in df.columns else df
        defaults = trending_df.head(8).to_dict(orient='records')
        for d in defaults:
            title_text = d.get('title') or d.get('name') or 'Unknown Title'
            d['title'] = title_text
            d['poster_url'] = get_cached_poster(client, d[id_col], media_type)
            d['watch_link'] = client.get_streaming_or_theatre_links(title_text, media_type, False)
        return defaults

    watchlist_indices = df[df[id_col].isin(user_watchlist_ids)].index.tolist()

    # A saved title is an interest signal, not necessarily a favourite.  When
    # available, explicit ratings and recency make the user's taste vector more
    # representative than a plain average of every saved title.
    source_weights = np.ones(len(watchlist_indices), dtype=float)
    excluded_ids = set(user_watchlist_ids)
    if user is not None:
        watchlist_rows = list(
            MovieWatchlist.objects.filter(user=user, media_type=media_type)
            .order_by('-added_at')
            .values('media_id', 'added_at')
        )
        rating_by_id = {
            row['media_id']: row['rating']
            for row in UserReview.objects.filter(user=user, media_type=media_type)
            .values('media_id', 'rating')
        }
        if media_type == 'movie':
            rating_by_id.update({
                row['movie_id']: row['rating']
                for row in Review.objects.filter(user=user).values('movie_id', 'rating')
            })
            excluded_ids.update(
                WatchedHistory.objects.filter(user=user).values_list('movie_id', flat=True)
            )

        excluded_ids.update(
            RecommendationFeedback.objects.filter(user=user, media_type=media_type)
            .values_list('media_id', flat=True)
        )

        weights_by_id = {}
        for rank, item in enumerate(watchlist_rows):
            # Recent activity decays smoothly instead of discarding older taste.
            recency_weight = 0.85 ** rank
            rating = rating_by_id.get(item['media_id'])
            if rating is None:
                rating_weight = 0.60
            else:
                max_rating = 10.0 if media_type == 'movie' and rating > 5 else 5.0
                rating_weight = max(0.15, min(1.0, float(rating) / max_rating))
            weights_by_id[item['media_id']] = (0.40 + 0.60 * recency_weight) * rating_weight

        source_weights = np.array(
            [weights_by_id.get(df.iloc[idx][id_col], 0.60) for idx in watchlist_indices],
            dtype=float,
        )

    # 2. COLD START LAYER 2: Watchlist items not found in DataFrame -> Genre Fallback
    if not watchlist_indices:
        print("[ML ENGINE] Fallback Triggered: Watchlist item not found in vector index.")
        recent_id = user_watchlist_ids[0] if user_watchlist_ids else None
        return get_genre_fallback_recommendations(recent_id, df, id_col, watchlist_indices, media_type, client)

    try:
        # Weighted centroid: recent, highly-rated saves carry more influence.
        mean_sim = np.average(sim_matrix[watchlist_indices], axis=0, weights=source_weights)

        candidate_indices = [
            idx for idx in range(len(df))
            if idx not in watchlist_indices and df.iloc[idx][id_col] not in excluded_ids
        ]
        sorted_candidates = sorted(candidate_indices, key=lambda idx: mean_sim[idx], reverse=True)

        # Set Cosine Similarity Threshold (tuned to 0.25 for sparse BoW features)
        threshold = 0.25
        high_sim_candidates = [idx for idx in sorted_candidates if mean_sim[idx] > threshold]

        top_3_scores = np.sort(mean_sim)[::-1][:3]
        print(f"\n[ML ENGINE] Input Watchlist ID(s): {user_watchlist_ids}")
        print(f"[ML ENGINE] Top 3 similarity scores: {list(top_3_scores)}")

        if len(top_3_scores) == 0 or all(score == 0.0 for score in top_3_scores):
            recent_id = user_watchlist_ids[0] if user_watchlist_ids else None
            return get_genre_fallback_recommendations(recent_id, df, id_col, watchlist_indices, media_type, client)

        # Maximal marginal relevance keeps the row useful: it favours relevant
        # titles while avoiding eight near-identical recommendations.  Start
        # from a wider relevance pool so diversity has meaningful alternatives.
        pool = sorted_candidates[:250]
        top_indices = []
        seen_franchises = set()
        diversity_strength = 0.35

        def franchise_key(index):
            """A lightweight guard against a row being dominated by sequels."""
            title = str(df.iloc[index].get('title') or df.iloc[index].get('name') or '')
            words = [word.lower() for word in re.findall(r"[a-zA-Z]+", title)]
            words = [word for word in words if word not in {'the', 'a', 'an', 'of', 'and', 'in'}]
            return ' '.join(words[:2]) if len(words) >= 2 else ''

        while pool and len(top_indices) < 8:
            eligible_pool = [idx for idx in pool if franchise_key(idx) not in seen_franchises]
            if not eligible_pool:
                break
            if not top_indices:
                best_idx = eligible_pool[0]
            else:
                best_idx = max(
                    eligible_pool,
                    key=lambda idx: (
                        (1.0 - diversity_strength) * mean_sim[idx]
                        - diversity_strength * max(sim_matrix[idx][chosen] for chosen in top_indices)
                    ),
                )
            top_indices.append(best_idx)
            pool.remove(best_idx)
            key = franchise_key(best_idx)
            if key:
                seen_franchises.add(key)

        recommendations = df.iloc[top_indices].to_dict(orient='records')

        for rec in recommendations:
            media_id = rec[id_col]
            title_text = rec.get('title') or rec.get('name') or 'Unknown Title'
            rec['title'] = title_text
            rec['poster_url'] = get_cached_poster(client, media_id, media_type)
            rec['watch_link'] = client.get_streaming_or_theatre_links(title_text, media_type, False)
            # Explain the strongest matching taste signal on the card.
            source_pos = int(np.argmax([sim_matrix[source_idx][df.index[df[id_col] == media_id][0]] * source_weights[pos]
                                        for pos, source_idx in enumerate(watchlist_indices)]))
            source = df.iloc[watchlist_indices[source_pos]]
            source_title = source.get('title') or source.get('name') or 'your watchlist'
            rec['recommendation_reason'] = f"Because you saved {source_title}"

        return recommendations

    except Exception as e:
        print(f"[ML INFERENCE] Error during recommendation calculation: {e}")
        return []

# ======================================================================
# User Authentication, Registration, and Home Views
# ======================================================================
from django.contrib import messages
from django.contrib.auth import logout, authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.views.decorators.cache import never_cache

@csrf_protect
@never_cache
def signup_view(request):
    if request.user.is_authenticated:
        return redirect('for_you_feed')

    from core.utils import fetch_tmdb_catalog
    catalog_p1 = fetch_tmdb_catalog(endpoint_type="movie", list_type="popular", page=1)
    catalog_p2 = fetch_tmdb_catalog(endpoint_type="movie", list_type="popular", page=2)
    movies_list = catalog_p1.get('results', []) + catalog_p2.get('results', [])
    movies_list = movies_list[:24]
    random_posters = [m['poster_url'] for m in movies_list if m.get('poster_url')]

    if request.method == 'POST':
        form = CineMatchRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            UserProfile.objects.get_or_create(user=user)
            login(request, user)
            messages.success(request, f"Welcome to CineMatch, {user.username}! Your account was created successfully.")
            return redirect('for_you_feed')
        else:
            messages.error(request, "Please correct the registration errors below.")
    else:
        form = CineMatchRegistrationForm()
        
    return render(request, 'core/signup.html', {
        'form': form,
        'random_posters': random_posters,
    })

# Maintain register_user as a compatibility alias for signup_view
register_user = signup_view

@csrf_protect
@never_cache
def login_view(request):
    if request.user.is_authenticated:
        return redirect('for_you_feed')

    from core.utils import fetch_tmdb_catalog
    catalog_p1 = fetch_tmdb_catalog(endpoint_type="movie", list_type="popular", page=1)
    catalog_p2 = fetch_tmdb_catalog(endpoint_type="movie", list_type="popular", page=2)
    movies_list = catalog_p1.get('results', []) + catalog_p2.get('results', [])
    movies_list = movies_list[:24]
    random_posters = [m['poster_url'] for m in movies_list if m.get('poster_url')]

    if request.method == 'POST':
        form = CineMatchLoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {username}!")
                next_url = request.GET.get('next')
                if next_url:
                    return redirect(next_url)
                return redirect('for_you_feed')
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = CineMatchLoginForm()
        
    return render(request, 'core/login.html', {
        'form': form,
        'random_posters': random_posters,
    })

def logout_view(request):
    logout(request)
    messages.info(request, "You have been logged out successfully.")
    return redirect('login')

def home_redirect(request):
    """
    Intelligent Root Redirection gatekeeper (Unit 8 Routing)
    Redirects all users to the home feed.
    """
    return redirect('for_you_feed')



# ======================================================================
# Watchlist CRUD: Add Item View (Create)
# ======================================================================
@login_required
@require_POST
def watchlist_add(request):
    media_id = request.POST.get('media_id')
    media_type = request.POST.get('media_type', 'movie')

    if not media_id:
        return JsonResponse({'success': False, 'error': 'Missing media_id parameter.'}, status=400)
    
    if media_type not in ['movie', 'tv']:
        return JsonResponse({'success': False, 'error': 'Invalid media_type parameter.'}, status=400)

    try:
        media_id = int(media_id)
        watchlist_item, created = MovieWatchlist.objects.get_or_create(
            user=request.user,
            media_id=media_id,
            media_type=media_type
        )
        
        # Invalidate recommendation cache
        from django.core.cache import cache
        try:
            cache.delete(f'user_feed_{request.user.id}')
        except Exception as ce:
            print(f"[CACHE ERROR] Invalidation failed in watchlist_add: {ce}")

        if created:
            return JsonResponse({'success': True, 'message': 'Title successfully added to watchlist.'})
        else:
            return JsonResponse({'success': True, 'message': 'Title is already in your watchlist.'})
    except ValueError:
        return JsonResponse({'success': False, 'error': 'media_id must be a valid integer.'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

# ======================================================================
# Watchlist CRUD: Delete Item View (Delete)
# ======================================================================
@login_required
@require_POST
def watchlist_delete(request):
    media_id = request.POST.get('media_id')
    media_type = request.POST.get('media_type', 'movie')

    if not media_id:
        return JsonResponse({'success': False, 'error': 'Missing media_id parameter.'}, status=400)

    try:
        media_id = int(media_id)
        deleted_count, _ = MovieWatchlist.objects.filter(
            user=request.user,
            media_id=media_id,
            media_type=media_type
        ).delete()

        # Invalidate recommendation cache
        from django.core.cache import cache
        try:
            cache.delete(f'user_feed_{request.user.id}')
        except Exception as ce:
            print(f"[CACHE ERROR] Invalidation failed in watchlist_delete: {ce}")

        if deleted_count > 0:
            return JsonResponse({'success': True, 'message': 'Title successfully removed from watchlist.'})
        else:
            return JsonResponse({'success': False, 'error': 'Title not found in your watchlist.'}, status=404)
    except ValueError:
        return JsonResponse({'success': False, 'error': 'media_id must be a valid integer.'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@login_required
@require_POST
def mark_not_interested(request):
    """Hide a recommendation and remove it from the next personalised feed."""
    media_id = request.POST.get('media_id')
    media_type = request.POST.get('media_type', 'movie')
    if media_type not in ('movie', 'tv'):
        return JsonResponse({'success': False, 'error': 'Invalid media_type parameter.'}, status=400)
    try:
        feedback, created = RecommendationFeedback.objects.get_or_create(
            user=request.user,
            media_id=int(media_id),
            media_type=media_type,
        )
    except (TypeError, ValueError):
        return JsonResponse({'success': False, 'error': 'media_id must be a valid integer.'}, status=400)

    from django.core.cache import cache
    cache.delete(f'user_feed_{request.user.id}')
    message = 'We will show fewer recommendations like this.' if created else 'This title is already hidden from your recommendations.'
    return JsonResponse({'success': True, 'message': message})

def get_personalized_recommendations(user):
    """
    Syllabus Topic: Recommendation logic encapsulation (Unit 8)
    Fetches personalized movie and TV show recommendations based on user's watchlist.
    """
    watchlist_items = MovieWatchlist.objects.filter(user=user)
    saved_movies = list(watchlist_items.filter(media_type='movie').values_list('media_id', flat=True))
    saved_tv_shows = list(watchlist_items.filter(media_type='tv').values_list('media_id', flat=True))
    
    recommended_movies = get_recommendations(saved_movies, 'movie', user=user)
    recommended_tv_shows = get_recommendations(saved_tv_shows, 'tv', user=user)
    
    return {
        'recommended_movies': recommended_movies,
        'recommended_tv_shows': recommended_tv_shows
    }

# ======================================================================
# Home / "For You" Feed Loader (Read & Recommend)
# ======================================================================
def get_tmdb_trailer_key(media_id, media_type):
    from .tmdb_api import TMDBClient
    client = TMDBClient()
    url = f"{client.base_url}/{media_type}/{media_id}/videos"
    try:
        response = get_resilient_session().get(url, headers=client.headers, timeout=5.0)
        if response.status_code == 200:
            results = response.json().get('results', [])
            youtube_vids = [r for r in results if r.get('site') == 'YouTube']
            if not youtube_vids:
                return None
            
            def score_video(vid):
                name = vid.get('name', '').lower()
                vtype = vid.get('type', '').lower()
                score = 0
                
                if vid.get('official'):
                    score += 10
                
                if vtype == 'trailer':
                    score += 50
                elif vtype == 'teaser':
                    score += 30
                elif vtype == 'promo':
                    score += 20
                
                if 'trailer' in name:
                    score += 40
                if 'official' in name:
                    score += 15
                if 'teaser' in name:
                    score += 20
                if 'promo' in name:
                    score += 10
                if 'season' in name or 'series' in name:
                    score += 5
                
                # Heavy penalty for bloopers, behind the scenes, interviews, etc.
                penalties = ['blooper', 'behind the scene', 'interview', 'bts', 'cast', 'clip', 'scene', 'featurette', 'preview clip', 'review', 'deleted scene']
                for penalty in penalties:
                    if penalty in name or penalty in vtype:
                        score -= 100
                
                return score
            
            youtube_vids.sort(key=score_video, reverse=True)
            best_vid = youtube_vids[0]
            if score_video(best_vid) > -50:
                return best_vid.get('key')
    except Exception as e:
        print(f"[SPOTLIGHT TRAILER SEARCH ERROR] ID {media_id}: {e}")
    return None

def for_you_feed(request):
    user = request.user
    client = TMDBClient()
    
    if user.is_authenticated:
        watchlist_items = MovieWatchlist.objects.filter(user=user)
        saved_movies = list(watchlist_items.filter(media_type='movie').values_list('media_id', flat=True))
        saved_tv_shows = list(watchlist_items.filter(media_type='tv').values_list('media_id', flat=True))
        saved_ids = list(watchlist_items.values_list('media_id', flat=True))
        watchlist_count = watchlist_items.count()
    else:
        saved_movies = []
        saved_tv_shows = []
        saved_ids = []
        watchlist_count = 0
    
    spotlight_movies = [
        {
            'movie_id': 157336,
            'title': 'Interstellar',
            'overview': 'The adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances involved in an interstellar voyage.',
            'backdrop_path': '/static/images/image_4721ff.jpg',
            'backdrop_url': '/static/images/image_4721ff.jpg',
            'media_type': 'movie'
        },
        {
            'movie_id': 27205,
            'title': 'Inception',
            'overview': 'Cobb, a skilled thief who is absolute best in the dangerous art of extraction, steals valuable secrets from deep within the subconscious during the dream state.',
            'backdrop_path': '/static/images/inception_bg.jpg',
            'backdrop_url': '/static/images/inception_bg.jpg',
            'media_type': 'movie'
        },
        {
            'movie_id': 66732,
            'title': 'Stranger Things',
            'overview': 'When a young boy vanishes, a small town uncovers a mystery involving secret experiments, terrifying supernatural forces and one strange little girl.',
            'backdrop_path': '/static/images/stranger_things_bg.jpg',
            'backdrop_url': '/static/images/stranger_things_bg.jpg',
            'media_type': 'tv'
        },
        {
            'movie_id': 1396,
            'title': 'Breaking Bad',
            'overview': "A high school chemistry teacher diagnosed with inoperable lung cancer turns to manufacturing and selling methamphetamine with a former student in order to secure his family's future.",
            'backdrop_path': '/static/images/breaking_bad_bg.jpg',
            'backdrop_url': '/static/images/breaking_bad_bg.jpg',
            'media_type': 'tv'
        },
        {
            'movie_id': 60059,
            'title': 'Better Call Saul',
            'overview': 'Six years before Saul Goodman meets Walter White, we meet him when the man who will become Saul Goodman is known as Jimmy McGill, a small-time lawyer searching for his destiny, and, more immediately, hustling to make ends meet.',
            'backdrop_path': '/static/images/better_call_saul_bg.jpg',
            'backdrop_url': '/static/images/better_call_saul_bg.jpg',
            'media_type': 'tv'
        }
    ]
    
    from django.core.cache import cache
    for m in spotlight_movies:
        cache_key = f"spotlight_trailer_v2_{m['media_type']}_{m['movie_id']}"
        key = cache.get(cache_key)
        if not key:
            key = get_tmdb_trailer_key(m['movie_id'], m['media_type'])
            if key:
                cache.set(cache_key, key, 86400)
        m['trailer_key'] = key
        
    spotlight_movie = spotlight_movies[0]
    
    now_showing = []
    url = f"{client.base_url}/movie/now_playing?language=en-US&region=IN&page=1"
    
    try:
        response = get_resilient_session().get(url, headers=client.headers, timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            for movie_data in results:
                movie_id = movie_data.get('id')
                title = movie_data.get('title')
                if not movie_id or not title:
                    continue
                
                genre_ids = movie_data.get('genre_ids', [])
                genre_names = [TMDB_GENRE_MAP.get(gid) for gid in genre_ids if TMDB_GENRE_MAP.get(gid)]
                genres_str = " | ".join(genre_names[:2]) or "Drama"
                
                poster_url = get_cached_poster(client, movie_id, 'movie')
                encoded_title = urllib.parse.quote_plus(title)
                booking_url = f"https://in.bookmyshow.com/explore/home/ahmedabad?search={encoded_title}"
                trailer_url = f"https://www.youtube.com/results?search_query={encoded_title}+official+trailer"
                
                now_showing.append({
                    'media_id': movie_id,
                    'title': title,
                    'media_type': 'movie',
                    'genres': genres_str,
                    'poster_url': poster_url,
                    'booking_url': booking_url,
                    'trailer_url': trailer_url
                })
                
                if len(now_showing) >= 15:
                    break
    except Exception as e:
        print(f"[TMDB LIVE INGESTION] Error fetching live now playing: {e}")
        
    if not now_showing:
        now_showing_fallback = [
            {'media_id': 19995, 'title': 'Avatar', 'media_type': 'movie', 'genres': 'Action | Sci-Fi'},
            {'media_id': 49026, 'title': 'The Dark Knight Rises', 'media_type': 'movie', 'genres': 'Action | Thriller'},
            {'media_id': 119051, 'title': 'Wednesday', 'media_type': 'tv', 'genres': 'Mystery | Comedy'},
            {'media_id': 66732, 'title': 'Stranger Things', 'media_type': 'tv', 'genres': 'Sci-Fi | Fantasy'},
            {'media_id': 27205, 'title': 'Inception', 'media_type': 'movie', 'genres': 'Sci-Fi | Action'}
        ]
        for item in now_showing_fallback:
            item['poster_url'] = get_cached_poster(client, item['media_id'], item['media_type'])
            encoded_title = urllib.parse.quote_plus(item['title'])
            item['booking_url'] = f"https://in.bookmyshow.com/explore/home/ahmedabad?search={encoded_title}"
            item['trailer_url'] = f"https://www.youtube.com/results?search_query={encoded_title}+official+trailer"
            now_showing.append(item)
    
    # ── CACHE-ASIDE PATTERN FOR PERSONALIZED RECOMMENDATIONS ──
    from django.core.cache import cache
    if user.is_authenticated:
        cache_key = f"user_feed_{user.id}"
        recs = None
        try:
            recs = cache.get(cache_key)
            # Auto-invalidate stale empty cache (e.g. after a deploy fix)
            if recs is not None:
                if not recs.get('recommended_movies') and not recs.get('recommended_tv_shows'):
                    cache.delete(cache_key)
                    recs = None
        except Exception as ce:
            print(f"[CACHE ERROR] Failed to fetch feed cache: {ce}")

        if recs is None:
            try:
                recs = get_personalized_recommendations(user)
                cache.set(cache_key, recs, 1800)
            except Exception as re_err:
                print(f"[RECOMMENDATION ENGINE ERROR] Failed to get recommendations: {re_err}")
                recs = {
                    'recommended_movies': [],
                    'recommended_tv_shows': []
                }
    else:
        cache_key = "user_feed_anonymous"
        recs = None
        try:
            recs = cache.get(cache_key)
        except Exception as ce:
            print(f"[CACHE ERROR] Failed to fetch anonymous feed cache: {ce}")
        if recs is None:
            recs = {
                'recommended_movies': get_recommendations([], 'movie'),
                'recommended_tv_shows': get_recommendations([], 'tv')
            }
            try:
                cache.set(cache_key, recs, 3600)
            except Exception as ce:
                print(f"[CACHE ERROR] Failed to write anonymous feed cache: {ce}")

            
    recommended_movies = recs.get('recommended_movies', [])
    recommended_tv_shows = recs.get('recommended_tv_shows', [])

    # ── FETCH DAILY TRENDING MOVIES FROM TMDB WITH CACHE-ASIDE ──
    from core.utils import get_daily_trending_movies
    daily_trending, trending_last_updated = get_daily_trending_movies()

    if not daily_trending:
        defaults_list = [
            {'id': 27205, 'title': 'Inception', 'year': '2010', 'release_date': '2010-07-16'},
            {'id': 157336, 'title': 'Interstellar', 'year': '2014', 'release_date': '2014-11-07'},
            {'id': 19995, 'title': 'Avatar', 'year': '2009', 'release_date': '2009-12-18'},
            {'id': 49026, 'title': 'The Dark Knight Rises', 'year': '2012', 'release_date': '2012-07-20'},
            {'id': 24428, 'title': 'The Avengers', 'year': '2012', 'release_date': '2012-05-04'}
        ]
        daily_trending = []
        for item in defaults_list:
            daily_trending.append({
                'id': item['id'],
                'title': item['title'],
                'poster_url': get_cached_poster(client, item['id'], 'movie'),
                'year': item['year'],
                'release_date': item['release_date']
            })

    talk_of_town = daily_trending[:3]
    most_interested = daily_trending[:5]
    
    # ── FETCH UPCOMING MOVIES FROM TMDB WITH CACHE-ASIDE (24H CACHE) ──
    from core.utils import get_upcoming_movies
    upcoming_movies = get_upcoming_movies()[:6]
    
    # ── Platform-specific feeds ──
    import random
    
    # Netflix (8)
    netflix_movies = get_provider_recommendations(8, 'movie')[:5]
    netflix_tv = get_provider_recommendations(8, 'tv')[:5]
    netflix_data = netflix_movies + netflix_tv
    random.shuffle(netflix_data)
    for film in netflix_data:
        if 'title' not in film: film['title'] = film.get('name', 'N/A')
        if 'name' not in film: film['name'] = film.get('title', 'N/A')
    
    # Prime (119)
    prime_movies = get_provider_recommendations(119, 'movie')[:5]
    prime_tv = get_provider_recommendations(119, 'tv')[:5]
    prime_data = prime_movies + prime_tv
    random.shuffle(prime_data)
    for film in prime_data:
        if 'title' not in film: film['title'] = film.get('name', 'N/A')
        if 'name' not in film: film['name'] = film.get('title', 'N/A')
    
    # Apple TV (350)
    apple_tv_movies = get_provider_recommendations(350, 'movie')[:5]
    apple_tv_tv = get_provider_recommendations(350, 'tv')[:5]
    apple_tv_data = apple_tv_movies + apple_tv_tv
    random.shuffle(apple_tv_data)
    for film in apple_tv_data:
        if 'title' not in film: film['title'] = film.get('name', 'N/A')
        if 'name' not in film: film['name'] = film.get('title', 'N/A')
    
    # ── DEBUG LOGGING FOR USER QUERY VERIFICATION ──
    print(f"[DEBUG] netflix_movies count: {len(netflix_data)}")
    if not netflix_data:
        print("[WARNING] netflix_movies query returned 0 rows or variable is empty!")
    else:
        print(f"[DEBUG] First entry of netflix_movies: {netflix_data[0]}")
        
    context = {
        'watchlist_count': watchlist_count,
        'saved_movies': saved_movies,
        'saved_tv_shows': saved_tv_shows,
        'saved_ids': saved_ids,
        'now_showing': now_showing,
        'recommended_movies': recommended_movies,
        'recommended_tv_shows': recommended_tv_shows,
        'spotlight_movie': spotlight_movie,
        'spotlight_movies': spotlight_movies,
        'talk_of_town': talk_of_town,
        'most_interested': most_interested,
        'trending_last_updated': trending_last_updated,
        'upcoming_movies': upcoming_movies,
        # 💎 INJECTED INTO CONTEXT
        'netflix_movies': netflix_data,
        'prime_movies': prime_data,
        'apple_tv_movies': apple_tv_data,
    }
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.GET.get('format') == 'json':
        return JsonResponse({
            'success': True,
            'watchlist_count': context['watchlist_count'],
            'saved_movies': context['saved_movies'],
            'saved_tv_shows': context['saved_tv_shows'],
            'saved_ids': saved_ids,
            'now_showing': now_showing,
            'recommended_movies': recommended_movies,
            'recommended_tv_shows': recommended_tv_shows,
            'spotlight_movie': spotlight_movie,
            'talk_of_town': talk_of_town,
            'most_interested': most_interested,
            'upcoming_movies': upcoming_movies,
            # 💎 INJECTED INTO JSON
            'netflix_movies': netflix_data,
            'prime_movies': prime_data,
            'apple_tv_movies': apple_tv_data,
        })
        
    return render(request, 'core/for_you.html', context)
# ======================================================================
# Explore Movies View
# ======================================================================
def explore_movies(request):
    from core.utils import fetch_tmdb_catalog
    
    query = request.GET.get('q', '').strip()
    page_number = request.GET.get('page', 1)
    try:
        page_number = int(page_number)
    except ValueError:
        page_number = 1
        
    api_response = fetch_tmdb_catalog(endpoint_type="movie", list_type="popular", query=query, page=page_number)
    movies_records = api_response.get('results', [])
    total_pages = min(api_response.get('total_pages', 1), 500)
    
    # Simulate page obj mapping for template pagination compatibility
    class MockPage:
        def __init__(self, number, object_list, max_pages):
            self.number = number
            self.object_list = object_list
            self.has_previous = number > 1
            self.previous_page_number = number - 1
            self.has_next = number < max_pages
            self.next_page_number = number + 1
            self.has_other_pages = max_pages > 1

    class MockPaginator:
        def __init__(self, max_pages):
            self.max_pages = max_pages

        @property
        def num_pages(self):
            return self.max_pages

    page_obj = MockPage(page_number, movies_records, total_pages)
    page_obj.paginator = MockPaginator(total_pages)

    # Check if request is AJAX
    is_ajax = request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.GET.get('ajax') == 'true'

    if request.user.is_authenticated:
        watchlist_ids = list(MovieWatchlist.objects.filter(
            user=request.user, media_type='movie'
        ).values_list('media_id', flat=True))
    else:
        watchlist_ids = []

    context = {
        'movies':       movies_records,
        'page_obj':     page_obj,
        'watchlist_ids': watchlist_ids,
        'query':         query,
    }

    if is_ajax:
        return render(request, 'core/includes/movie_grid_partial.html', context)

    return render(request, 'core/explore_movies.html', context)

# ======================================================================
# Explore TV Shows View
# ======================================================================
def explore_tv(request):
    from core.utils import fetch_tmdb_catalog
    
    query = request.GET.get('q', '').strip()
    page_number = request.GET.get('page', 1)
    try:
        page_number = int(page_number)
    except ValueError:
        page_number = 1
        
    api_response = fetch_tmdb_catalog(endpoint_type="tv", list_type="popular", query=query, page=page_number)
    tv_records = api_response.get('results', [])
    total_pages = min(api_response.get('total_pages', 1), 500)
    
    # Simulate page obj mapping for template pagination compatibility
    class MockPage:
        def __init__(self, number, object_list, max_pages):
            self.number = number
            self.object_list = object_list
            self.has_previous = number > 1
            self.previous_page_number = number - 1
            self.has_next = number < max_pages
            self.next_page_number = number + 1
            self.has_other_pages = max_pages > 1

    class MockPaginator:
        def __init__(self, max_pages):
            self.max_pages = max_pages

        @property
        def num_pages(self):
            return self.max_pages

    page_obj = MockPage(page_number, tv_records, total_pages)
    page_obj.paginator = MockPaginator(total_pages)

    # Check if request is AJAX
    is_ajax = request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.GET.get('ajax') == 'true'

    if request.user.is_authenticated:
        watchlist_ids = list(MovieWatchlist.objects.filter(
            user=request.user, media_type='tv'
        ).values_list('media_id', flat=True))
    else:
        watchlist_ids = []

    context = {
        'tv_shows':      tv_records,
        'page_obj':      page_obj,
        'watchlist_ids': watchlist_ids,
        'query':         query,
    }

    if is_ajax:
        return render(request, 'core/includes/tv_grid_partial.html', context)

    return render(request, 'core/explore_tv.html', context)

# ======================================================================
# User History Statistics & Persona Aggregation Helpers
# ======================================================================
from django.db.models import Sum, Avg

def get_user_stats(user):
    """
    Queries WatchedHistory database model to calculate watchtime, rating, and genre counts.
    Seeds mock entries if user has no prior history.
    """
    history = WatchedHistory.objects.filter(user=user)
    
    # If user has no watch history, populate mock records for instant visualizations
    if history.count() == 0:
        mock_data = [
            {"movie_id": 299534, "movie_title": "Avengers: Endgame", "duration": 181, "rating": 9.0, "genres": "Action, Adventure, Sci-Fi"},
            {"movie_id": 550, "movie_title": "Fight Club", "duration": 139, "rating": 8.5, "genres": "Drama"},
            {"movie_id": 155, "movie_title": "The Dark Knight", "duration": 152, "rating": 9.5, "genres": "Action, Crime, Drama"},
            {"movie_id": 680, "movie_title": "Pulp Fiction", "duration": 154, "rating": 8.0, "genres": "Crime, Thriller"},
            {"movie_id": 13, "movie_title": "Forrest Gump", "duration": 142, "rating": 7.5, "genres": "Comedy, Drama, Romance"},
            {"movie_id": 27205, "movie_title": "Inception", "duration": 148, "rating": 8.8, "genres": "Action, Sci-Fi, Thriller"},
            {"movie_id": 120, "movie_title": "The Lord of the Rings: The Fellowship of the Ring", "duration": 178, "rating": 9.2, "genres": "Action, Adventure, Fantasy"},
        ]
        for item in mock_data:
            WatchedHistory.objects.create(
                user=user,
                movie_id=item["movie_id"],
                movie_title=item["movie_title"],
                duration=item["duration"],
                rating=item["rating"],
                genres=item["genres"]
            )
        history = WatchedHistory.objects.filter(user=user)
    
    agg = history.aggregate(total_time=Sum('duration'), avg_rating=Avg('rating'))
    total_time = agg['total_time'] or 0
    avg_rating = agg['avg_rating'] or 0.0
    
    genre_counts = {}
    for entry in history:
        if entry.genres:
            for g in entry.genres.split(','):
                g_clean = g.strip()
                if g_clean:
                    genre_counts[g_clean] = genre_counts.get(g_clean, 0) + 1
                    
    return {
        'total_watchtime_mins': total_time,
        'total_watchtime_hours': round(total_time / 60.0, 1),
        'genre_distribution': genre_counts,
        'avg_rating': round(avg_rating, 1),
        'movies_watched_count': history.count()
    }

def get_user_persona(stats):
    """
    Evaluates statistics metrics and assigns a user profile label persona.
    """
    hours = stats.get('total_watchtime_hours', 0)
    avg_rating = stats.get('avg_rating', 0.0)
    count = stats.get('movies_watched_count', 0)
    
    if hours > 100:
        return {
            'title': 'The Binge-Watcher',
            'desc': 'You devour movies in massive quantities. Sleeping is optional; the cinema is your home.',
            'icon': 'fa-tv'
        }
    elif avg_rating > 8.5:
        return {
            'title': 'The Optimistic Fanatic',
            'desc': 'You love everything you watch! Every movie is a masterpiece in your eyes.',
            'icon': 'fa-heart'
        }
    elif avg_rating < 5.0 and avg_rating > 0:
        return {
            'title': 'The Critical Cinephile',
            'desc': 'Hard to please. You spot flaws in scripts, editing, and cinematography with ease.',
            'icon': 'fa-magnifying-glass'
        }
    elif count > 5:
        return {
            'title': 'The Balanced Connoisseur',
            'desc': 'A diverse viewer who enjoys a good mix of drama, actions, and comedies.',
            'icon': 'fa-circle-check'
        }
    else:
        return {
            'title': 'The Casual Viewer',
            'desc': 'You watch movies selectively when the mood strikes.',
            'icon': 'fa-ticket'
        }

# ======================================================================
# Interactive Analytics Dashboard View
# ======================================================================
@login_required
def analytics_dashboard(request):
    import plotly.graph_objects as go
    import plotly.io as pio
    from .analytics_engine import generate_seaborn_heatmap, generate_plotly_scatter, generate_networkx_graph
    
    user = request.user
    watchlist_items = MovieWatchlist.objects.filter(user=user, media_type='movie')
    watchlist_movies = list(watchlist_items.values_list('media_id', flat=True))
    
    # ── CALCULATE STATS & PERSONA ──
    stats = get_user_stats(user)
    persona = get_user_persona(stats)
    
    # ── PLOTLY PIE CHART: GENRE WATCH DISTRIBUTION ──
    genres = list(stats['genre_distribution'].keys())
    counts = list(stats['genre_distribution'].values())
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=genres, 
        values=counts,
        hole=.3,
        marker=dict(colors=['#a855f7', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#8b5cf6'])
    )])
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#ffffff", 'family': "Inter"},
        margin=dict(l=20, r=20, t=20, b=20),
        legend={'font': {'color': '#ffffff', 'size': 11}}
    )
    pie_json = pio.to_json(fig_pie)
    
    # ── PLOTLY GAUGE: MEAN RATINGS ──
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = stats['avg_rating'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "#ffffff"},
            'bar': {'color': "#a855f7"},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 5], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [5, 8], 'color': 'rgba(234, 179, 8, 0.2)'},
                {'range': [8, 10], 'color': 'rgba(34, 197, 94, 0.2)'}
            ],
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#ffffff", 'family': "Inter"},
         margin=dict(l=30, r=30, t=50, b=20)
    )
    gauge_json = pio.to_json(fig_gauge)
    
    try:
        heatmap_base64 = generate_seaborn_heatmap()
    except Exception:
        heatmap_base64 = ""
    try:
        plotly_div_html = generate_plotly_scatter()
    except Exception:
        plotly_div_html = ""
    try:
        network_base64 = generate_networkx_graph(watchlist_movies)
    except Exception:
        network_base64 = ""
    
    context = {
        'heatmap_img': heatmap_base64,
        'plotly_div': plotly_div_html,
        'network_img': network_base64,
        'stats': stats,
        'persona': persona,
        'pie_json': pie_json,
        'gauge_json': gauge_json,
    }
    
    return render(request, 'core/analytics.html', context)


# ======================================================================
# Movie Detail Hub View
# ======================================================================
def movie_detail_view(request, movie_id):
    # ── ROUTE/TYPE DETECTION ──
    from django.shortcuts import redirect
    from core.models import CachedMedia
    try:
        movie_id_val = int(movie_id)
    except ValueError:
        movie_id_val = 0
    if request.GET.get('type') == 'tv' or CachedMedia.objects.filter(media_id=movie_id_val, media_type='tv').exists():
        return redirect('tv_show_detail', series_id=movie_id_val)

    # ── SESSION BASED RECENTLY VIEWED LOGIC ──
    recently_viewed = request.session.get('recently_viewed', [])
    try:
        recently_viewed = [item for item in recently_viewed if not (isinstance(item, dict) and item.get('id') == movie_id_val and item.get('type') == 'movie')]
        recently_viewed.insert(0, {'id': movie_id_val, 'type': 'movie'})
        request.session['recently_viewed'] = recently_viewed[:6]
    except Exception as e:
        print(f"[RECENTLY VIEWED] Session log failed: {e}")

    api_key = settings.TMDB_API_KEY
    endpoint = (
        f"https://api.themoviedb.org/3/movie/{movie_id}"
        f"?api_key={api_key}"
        f"&language=en-US"
        f"&append_to_response=credits,videos,watch/providers,similar,reviews"
    )

    movie = {}
    cast = []
    trailer_key = None
    trailer_url = None
    watch_providers = []
    similar_movies = []
    tmdb_reviews = []
    belongs_to_collection = None
    collection_movies = []
    collection_name = ""
    omdb_data = None
    # VIVA JUSTIFICATION / COLD START CACHING STRATEGY:
    # I implemented a Write-Through Caching pattern where the local database acts as the primary data store.
    # By integrating Just-in-Time (JIT) fetching, the system dynamically populates its cache upon the
    # first request for any media item, ensuring data availability while maintaining high performance for all subsequent hits.
    from core.models import CachedMedia
    is_manual_override = False
    manual_providers = []
    try:
        cached_record = CachedMedia.objects.get(media_id=movie_id_val, media_type='movie')
        data = cached_record.data
        is_manual_override = cached_record.is_manual_override
        manual_providers = cached_record.manual_providers or []
    except CachedMedia.DoesNotExist:
        data = None

    if not data:
        from django.core.cache import cache
        cache_key = f"movie_detail_data_{movie_id}"
        data = cache.get(cache_key)

    if not data:
        try:
            resp = get_resilient_session().get(endpoint, timeout=10.0)
            if resp.status_code == 404:
                tv_url = f"https://api.themoviedb.org/3/tv/{movie_id}?api_key={api_key}"
                tv_resp = get_resilient_session().get(tv_url, timeout=5.0)
                if tv_resp.status_code == 200:
                    return redirect('tv_show_detail', series_id=movie_id_val)
            resp.raise_for_status()
            data = resp.json()
            # Write-through cache: save to both Django cache and local database cache
            from django.core.cache import cache
            cache.set(f"movie_detail_data_{movie_id}", data, 86400)
            CachedMedia.objects.update_or_create(
                media_id=movie_id_val,
                media_type='movie',
                defaults={'data': data}
            )
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"[MOVIE DETAIL] TMDB API request failed for movie_id={movie_id}, fallback failed: {e}")
            data = None

    if data:
        # Data Enrichment Logic: explicitly query watch/providers if missing or incomplete in cached data
        if not is_manual_override:
            providers_payload = data.get('watch/providers', {}).get('results', {})
            if not providers_payload or (not providers_payload.get('IN') and not providers_payload.get('US')):
                try:
                    prov_url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers?api_key={api_key}"
                    prov_resp = get_resilient_session().get(prov_url, timeout=3.0)
                    if prov_resp.status_code == 200:
                        prov_data = prov_resp.json()
                        data['watch/providers'] = prov_data
                        # Write-through/enrich the cached payload in db
                        CachedMedia.objects.update_or_create(
                            media_id=movie_id_val,
                            media_type='movie',
                            defaults={'data': data}
                        )
                except Exception as pe:
                    print(f"[DATA ENRICHMENT WARNING] Failed to enrich watch/providers for movie ID {movie_id}: {pe}")

        # Data Enrichment Logic: explicitly query credits if missing or incomplete in cached data
        if not data.get('credits') or not data.get('credits', {}).get('cast'):
            try:
                credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={api_key}"
                credits_resp = get_resilient_session().get(credits_url, timeout=3.0)
                if credits_resp.status_code == 200:
                    credits_data = credits_resp.json()
                    data['credits'] = credits_data
                    # Write-through/enrich the cached payload in db
                    CachedMedia.objects.update_or_create(
                        media_id=movie_id_val,
                        media_type='movie',
                        defaults={'data': data}
                    )
            except Exception as ce:
                print(f"[DATA ENRICHMENT WARNING] Failed to enrich credits for movie ID {movie_id}: {ce}")
        # Data Enrichment Logic: explicitly query videos if missing or incomplete in cached data
        if not data.get('videos') or not data.get('videos', {}).get('results'):
            try:
                videos_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={api_key}"
                videos_resp = get_resilient_session().get(videos_url, timeout=3.0)
                if videos_resp.status_code == 200:
                    videos_data = videos_resp.json()
                    data['videos'] = videos_data
                    # Write-through/enrich the cached payload in db
                    CachedMedia.objects.update_or_create(
                        media_id=movie_id_val,
                        media_type='movie',
                        defaults={'data': data}
                    )
            except Exception as ve:
                print(f"[DATA ENRICHMENT WARNING] Failed to enrich videos for movie ID {movie_id}: {ve}")
        
        # Data Enrichment Logic: explicitly query reviews if missing in cached data
        if 'reviews' not in data:
            try:
                reviews_url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={api_key}"
                reviews_resp = get_resilient_session().get(reviews_url, timeout=3.0)
                if reviews_resp.status_code == 200:
                    reviews_data = reviews_resp.json()
                    data['reviews'] = reviews_data
                    # Write-through/enrich the cached payload in db & django cache
                    from django.core.cache import cache
                    cache.set(f"movie_detail_data_{movie_id}", data, 86400)
                    CachedMedia.objects.update_or_create(
                        media_id=movie_id_val,
                        media_type='movie',
                        defaults={'data': data}
                    )
            except Exception as re:
                print(f"[DATA ENRICHMENT WARNING] Failed to enrich reviews for movie ID {movie_id}: {re}")

        try:
            imdb_id = data.get('imdb_id')
            if imdb_id:
                from core.utils import fetch_omdb_data
                omdb_data = fetch_omdb_data(imdb_id)

            poster_path = data.get('poster_path') or ''
            backdrop_path = data.get('backdrop_path') or ''
            
            usd_budget = data.get('budget', 0)
            usd_revenue = data.get('revenue', 0)
            inr_conversion_rate = 95.21
            
            movie = {
                'id':            data.get('id', movie_id),
                'imdb_id':       data.get('imdb_id'),
                'title':         data.get('title', 'Unknown Title'),
                'overview':      data.get('overview', ''),
                'release_date':  data.get('release_date', ''),
                'runtime':       data.get('runtime', 0),
                'vote_average':  round(data.get('vote_average', 0.0), 1),
                'genres':        [g.get('name', '') for g in data.get('genres', [])],
                'tagline':       data.get('tagline', ''),
                'budget_inr':    int(usd_budget * inr_conversion_rate),
                'revenue_inr':   int(usd_revenue * inr_conversion_rate),
                'production_companies': [
                    {
                        'name': c.get('name'),
                        'logo_url': f"https://image.tmdb.org/t/p/w92{c.get('logo_path')}" if c.get('logo_path') else None
                    }
                    for c in data.get('production_companies', []) if c.get('name')
                ][:4],
                'languages':            [l.get('english_name') for l in data.get('spoken_languages', []) if l.get('english_name')],
                'poster_url': (
                    f"https://image.tmdb.org/t/p/w500{poster_path}"
                    if poster_path else
                    'https://images.unsplash.com/photo-1542204172-e7052809f852?q=80&w=400&auto=format&fit=cover'
                ),
                'backdrop_url': (
                    f"https://image.tmdb.org/t/p/original{backdrop_path}"
                    if backdrop_path else ''
                ),
            }

            credits_payload = data.get('credits', {})
            raw_cast = credits_payload.get('cast', [])

            # 1. Crew definitions
            crew = credits_payload.get('crew', []) 
            
            # 2. Now find the director safely
            director = next((member for member in crew if member.get('job') == 'Director'), None)

            for member in raw_cast[:6]:
                profile_path = member.get('profile_path') or ''
                cast.append({
                    'id':         member.get('id'),
                    'name':       member.get('name', ''),
                    'character':  member.get('character', ''),
                    'profile_url': (
                        f"https://image.tmdb.org/t/p/w185{profile_path}"
                        if profile_path else
                        'https://ui-avatars.com/api/?name=' + urllib.parse.quote_plus(member.get('name', 'Actor'))
                    ),
                })

            videos_payload = data.get('videos', {})
            raw_videos = videos_payload.get('results', [])
            print(f"[DEBUG] movie_detail_view raw videos response count: {len(raw_videos)}")
            for idx, video in enumerate(raw_videos):
                print(f"[DEBUG] Video #{idx}: name={video.get('name')}, site={video.get('site')}, type={video.get('type')}, key={video.get('key')}, official={video.get('official')}")
            for video in raw_videos:
                if (
                    video.get('site') == 'YouTube'
                    and video.get('type') == 'Trailer'
                    and video.get('official', False)
                ):
                    trailer_key = video.get('key')
                    break
            if not trailer_key:
                for video in raw_videos:
                    if video.get('site') == 'YouTube' and video.get('type') == 'Trailer':
                        trailer_key = video.get('key')
                        break
            print(f"[DEBUG] movie_detail_view extracted trailer_key: {trailer_key}")

            if trailer_key:
                trailer_url = f"https://www.youtube.com/embed/{trailer_key}"

            if is_manual_override:
                watch_providers = manual_providers
            else:
                providers_payload = data.get('watch/providers', {}).get('results', {})
                region_data = providers_payload.get('IN') or providers_payload.get('US') or {}
                raw_providers = region_data.get('flatrate', [])
                for p in raw_providers:
                    logo_path = p.get('logo_path') or ''
                    watch_providers.append({
                        'name':     p.get('provider_name', ''),
                        'logo_url': (
                            f"https://image.tmdb.org/t/p/w92{logo_path}"
                            if logo_path else ''
                        ),
                    })

            # --- TasteDive Recommendations Fetching (TasteDive API Integration) ---
            tastedive_recs = []
            tastedive_success = False
            movie_title = movie.get('title')
            
            if movie_title:
                tastedive_key = getattr(settings, 'TASTEDIVE_API_KEY', '')
                tastedive_url = "https://tastedive.com/api/similar"
                td_params = {
                    'q': movie_title,
                    'type': 'movie',
                    'info': 1,
                    'k': tastedive_key
                }
                try:
                    td_resp = get_resilient_session().get(tastedive_url, params=td_params, timeout=5.0)
                    if td_resp.status_code == 200:
                        td_data = td_resp.json()
                        raw_results = td_data.get('Similar', {}).get('Results', [])
                        for item in raw_results[:5]:
                            tastedive_recs.append({
                                'title': item.get('name', 'Unknown Title'),
                                'teaser': item.get('description', ''),
                                'wiki_url': item.get('wUrl', ''),
                                'youtube_url': item.get('yUrl', ''),
                                'youtube_id': item.get('yID', ''),
                            })
                        tastedive_success = True
                except Exception as te:
                    print(f"[TASTEDIVE ERROR] Failed to fetch TasteDive recommendations for '{movie_title}': {te}")

            if tastedive_success and tastedive_recs:
                tastedive_active = True
            else:
                tastedive_active = False
                # Fallback: Extract similar movies from TMDB payload
                similar_payload = data.get('similar', {})
                raw_similar = similar_payload.get('results', [])
                
                if not raw_similar:
                    try:
                        similar_endpoint = f"https://api.themoviedb.org/3/movie/{movie_id_val}/similar?api_key={api_key}&language=en-US"
                        sim_resp = get_resilient_session().get(similar_endpoint, timeout=3.0)
                        if sim_resp.status_code == 200:
                            similar_payload = sim_resp.json()
                            raw_similar = similar_payload.get('results', [])
                    except Exception as e:
                        print(f"[MOVIE DETAIL] Similar endpoint query failed: {e}")

                hidden_ids = []
                if request.user.is_authenticated:
                    hidden_ids = list(RecommendationFeedback.objects.filter(user=request.user, media_type='movie').values_list('media_id', flat=True))

                for s in raw_similar:
                    s_id = s.get('id')
                    if s_id in hidden_ids:
                        continue
                    s_poster = s.get('poster_path') or ''
                    similar_movies.append({
                        'id':           s_id,
                        'title':        s.get('title', 'Unknown'),
                        'vote_average': round(s.get('vote_average', 0.0), 1),
                        'poster_url': (
                            f"https://image.tmdb.org/t/p/w300{s_poster}"
                            if s_poster else
                            'https://images.unsplash.com/photo-1542204172-e7052809f852?q=80&w=400&auto=format&fit=cover'
                        ),
                    })
                    if len(similar_movies) >= 5:
                        break

            # Franchise / Collection Fetching
            belongs_to_collection = data.get('belongs_to_collection')
            if belongs_to_collection:
                collection_id = belongs_to_collection.get('id')
                collection_name = belongs_to_collection.get('name', '')
                collection_endpoint = f"https://api.themoviedb.org/3/collection/{collection_id}?api_key={api_key}&language=en-US"
                try:
                    col_resp = get_resilient_session().get(collection_endpoint, timeout=3.0)
                    if col_resp.status_code == 200:
                        col_data = col_resp.json()
                        parts = col_data.get('parts', [])
                        for part in parts:
                            part_id = part.get('id')
                            # Exclude current movie from franchise items
                            if str(part_id) != str(movie_id):
                                part_poster = part.get('poster_path') or ''
                                release_date = part.get('release_date', '')
                                year = release_date.split('-')[0] if release_date else 'N/A'
                                collection_movies.append({
                                    'id': part_id,
                                    'movie_id': part_id,
                                    'title': part.get('title', 'Unknown'),
                                    'vote_average': round(part.get('vote_average', 0.0), 1),
                                    'year': year,
                                    'poster_url': (
                                        f"https://image.tmdb.org/t/p/w300{part_poster}"
                                        if part_poster else
                                        'https://images.unsplash.com/photo-1542204172-e7052809f852?q=80&w=400&auto=format&fit=crop'
                                    ),
                                })
                except Exception as ce:
                    print(f"[COLLECTION ERROR] Failed to fetch collection details for {collection_id}: {ce}")

            # Parse TMDB user reviews
            raw_tmdb_reviews = data.get('reviews', {}).get('results', [])
            for r in raw_tmdb_reviews:
                author_details = r.get('author_details', {})
                avatar_path = author_details.get('avatar_path') or ''
                if avatar_path.startswith('/http') or avatar_path.startswith('http'):
                    avatar_url = avatar_path[1:] if avatar_path.startswith('/') else avatar_path
                elif avatar_path:
                    avatar_url = f"https://image.tmdb.org/t/p/w92{avatar_path}"
                else:
                    avatar_url = f"https://ui-avatars.com/api/?name={urllib.parse.quote_plus(r.get('author', 'Critic'))}"
                
                rating = author_details.get('rating')
                
                tmdb_reviews.append({
                    'author': r.get('author', 'Anonymous'),
                    'avatar_url': avatar_url,
                    'rating': rating,
                    'content': r.get('content', ''),
                    'created_at': r.get('created_at', '')[:10],
                })
        except Exception as e:
            print(f"[MOVIE DETAIL] Unexpected parsing error for movie_id={movie_id}: {e}")
            data = None

    if not data:
        # Fallback to local dict or placeholder
        local_title = "Content temporarily unavailable"
        if MOVIE_DICT:
            try:
                df = pd.DataFrame(MOVIE_DICT)
                match = df[df['movie_id'] == movie_id_val]
                if not match.empty:
                    local_title = match.iloc[0]['title']
            except Exception:
                pass
        movie = {
            'id': movie_id_val,
            'title': local_title,
            'overview': 'We are currently experiencing connection timeout issues with the external API database. This content is temporarily unavailable. Please try again later.',
            'release_date': '',
            'runtime': 0,
            'vote_average': 0.0,
            'genres': ['Temporarily Offline'],
            'tagline': 'Offline Mode Enabled',
            'poster_url': 'https://images.unsplash.com/photo-1542204172-e7052809f852?q=80&w=400&auto=format&fit=cover',
            'backdrop_url': '',
        }

    is_in_watchlist = False
    if request.user.is_authenticated:
        is_in_watchlist = MovieWatchlist.objects.filter(
            user=request.user, media_id=movie_id, media_type='movie'
        ).exists()

    from datetime import datetime
    release_date_str = movie.get('release_date', '')
    is_now_showing = False
    is_released = True

    if release_date_str:
        try:
            release_date = datetime.strptime(release_date_str, '%Y-%m-%d').date()
            current_date = datetime.now().date()
            if release_date <= current_date and (current_date - release_date).days <= 45:
                is_now_showing = True
            elif release_date > current_date:
                is_now_showing = True
                is_released = False
        except ValueError:
            pass

    # ── FETCH REVIEWS DATALAYER AND AUTHOR ACCESSORS ──
    raw_reviews = Review.objects.filter(movie_id=movie_id).select_related('user')
    reviews_list = []
    for r in raw_reviews:
        sentiment_label, sentiment_score = predict_sentiment(r.content)
        r.sentiment_label = sentiment_label
        r.sentiment_score = sentiment_score
        reviews_list.append(r)
        
    user_review = None
    if request.user.is_authenticated:
        for r in reviews_list:
            if r.user == request.user:
                user_review = r
                break

    # Fetch watchlist IDs for bookmark toggle states
    watchlist_ids = []
    if request.user.is_authenticated:
        watchlist_ids = list(MovieWatchlist.objects.filter(
            user=request.user, media_type='movie'
        ).values_list('media_id', flat=True))

    # Fetch streaming links from Local DB / Cache (Write-Through batch retrieval)
    streaming_links = None
    if data:
        streaming_links = data.get('streaming_links')

    if streaming_links is None:
        from core.utils import get_streaming_links
        movie_title = movie.get('title', '')
        streaming_links = get_streaming_links(movie_title)
        if data:
            data['streaming_links'] = streaming_links
            from core.models import CachedMedia
            CachedMedia.objects.filter(media_id=movie_id_val, media_type='movie').update(data=data)

    # Map streaming links directly to the watch_providers items with TMDB fallback
    from core.utils import normalize_name
    if watch_providers:
        for provider in watch_providers:
            provider_name = provider.get('name', '')
            norm_provider = normalize_name(provider_name)

            # Find matching link from Watchmode
            matched_url = None
            if streaming_links:
                for service_name, url in streaming_links.items():
                    norm_service = normalize_name(service_name)
                    if norm_provider == norm_service or norm_provider in norm_service or norm_service in norm_provider:
                        matched_url = url
                        break
            
            # Fallback to TMDB watch URL if no Watchmode link matches
            if not matched_url:
                matched_url = f"https://www.themoviedb.org/movie/{movie_id}/watch?locale=IN"
                
            provider['web_url'] = matched_url

    # Data Debugging: Log the raw where_to_watch data to the terminal
    print(f"[DEBUG] movie_detail_view where_to_watch: {watch_providers}")

    # ── SOURCE FALLBACK STORYLINE RECONCILIATION ──
    tmdb_overview = movie.get('overview', '') if movie else ''
    omdb_plot = omdb_data.get('full_plot', '') if omdb_data else ''

    if omdb_plot and omdb_plot != 'N/A':
        storyline = omdb_plot
    else:
        storyline = tmdb_overview

    context = {
        'movie':           movie,
        'tagline':         movie.get('tagline', ''),
        'storyline':       storyline,
        'cast':            cast,
        'trailer_key':     trailer_key,
        'trailer_url':     trailer_url,
        'watch_providers': watch_providers,
        'manual_providers': manual_providers,
        'streaming_links': streaming_links, # passed directly to context
        'similar_movies':  similar_movies,
        'recommendations': similar_movies, # mapped to recommendations
        'tastedive_recs':   tastedive_recs,
        'tastedive_active': tastedive_active,
        'is_in_watchlist': is_in_watchlist,
        'is_now_showing':  is_now_showing,
        'is_released':     is_released,
        # 💎 INJECTED REVIEWS DATA CONTEXTS
        'reviews':         reviews_list,
        'tmdb_reviews':    tmdb_reviews,
        'user_review':     user_review,
        'director':        director,
        # Franchise Groups
        'belongs_to_collection': belongs_to_collection,
        'collection_movies':     collection_movies,
        'collection_name':       collection_name,
        'watchlist_ids':         watchlist_ids,
        'omdb_data':             omdb_data,
    }

    return render(request, 'core/movie_detail.html', context)


# ======================================================================
# TV Show Detail Hub View
# ======================================================================
def tv_detail_view(request, series_id):
    # ── ROUTE/TYPE DETECTION ──
    from django.shortcuts import redirect
    from core.models import CachedMedia
    try:
        series_id_val = int(series_id)
    except ValueError:
        series_id_val = 0
    if request.GET.get('type') == 'movie' or CachedMedia.objects.filter(media_id=series_id_val, media_type='movie').exists():
        return redirect('movie_detail', movie_id=series_id_val)

    # ── SESSION BASED RECENTLY VIEWED LOGIC ──
    recently_viewed = request.session.get('recently_viewed', [])
    try:
        recently_viewed = [item for item in recently_viewed if not (isinstance(item, dict) and item.get('id') == series_id_val and item.get('type') == 'tv')]
        recently_viewed.insert(0, {'id': series_id_val, 'type': 'tv'})
        request.session['recently_viewed'] = recently_viewed[:6]
    except Exception as e:
        print(f"[RECENTLY VIEWED] Session log failed: {e}")

    api_key = settings.TMDB_API_KEY

    endpoint = (
        f"https://api.themoviedb.org/3/tv/{series_id}"
        f"?api_key={api_key}"
        f"&language=en-US"
        f"&append_to_response=credits,videos,watch/providers,similar,external_ids,reviews"
    )

    tv_show = {}
    cast = []
    trailer_key = None
    watch_providers = []
    similar_shows = []
    tastedive_recs = []
    omdb_data = None
    tmdb_reviews = []

    # VIVA JUSTIFICATION / COLD START CACHING STRATEGY:
    # I implemented a Write-Through Caching pattern where the local database acts as the primary data store.
    # By integrating Just-in-Time (JIT) fetching, the system dynamically populates its cache upon the
    # first request for any media item, ensuring data availability while maintaining high performance for all subsequent hits.
    from core.models import CachedMedia
    try:
        cached_record = CachedMedia.objects.get(media_id=series_id_val, media_type='tv')
        data = cached_record.data
    except CachedMedia.DoesNotExist:
        data = None

    if not data:
        from django.core.cache import cache
        cache_key = f"tv_detail_data_{series_id}"
        data = cache.get(cache_key)

    if not data:
        try:
            resp = get_resilient_session().get(endpoint, timeout=10.0)
            if resp.status_code == 404:
                movie_url = f"https://api.themoviedb.org/3/movie/{series_id}?api_key={api_key}"
                movie_resp = get_resilient_session().get(movie_url, timeout=5.0)
                if movie_resp.status_code == 200:
                    return redirect('movie_detail', movie_id=series_id_val)
            resp.raise_for_status()
            data = resp.json()
            # Write-through cache: save to both Django cache and local database cache
            from django.core.cache import cache
            cache.set(f"tv_detail_data_{series_id}", data, 86400)
            CachedMedia.objects.update_or_create(
                media_id=series_id_val,
                media_type='tv',
                defaults={'data': data}
            )
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"[TV DETAIL] TMDB API request failed for series_id={series_id}, fallback failed: {e}")
            data = None

    if data:
        # Data Enrichment Logic: explicitly query watch/providers if missing or incomplete in cached data
        providers_payload = data.get('watch/providers', {}).get('results', {})
        if not providers_payload or (not providers_payload.get('IN') and not providers_payload.get('US')):
            try:
                prov_url = f"https://api.themoviedb.org/3/tv/{series_id_val}/watch/providers?api_key={api_key}"
                prov_resp = get_resilient_session().get(prov_url, timeout=3.0)
                if prov_resp.status_code == 200:
                    prov_data = prov_resp.json()
                    data['watch/providers'] = prov_data
                    # Write-through/enrich the cached payload in db & django cache
                    from django.core.cache import cache
                    cache.set(f"tv_detail_data_{series_id}", data, 86400)
                    CachedMedia.objects.update_or_create(
                        media_id=series_id_val,
                        media_type='tv',
                        defaults={'data': data}
                    )
            except Exception as pe:
                print(f"[DATA ENRICHMENT WARNING] Failed to enrich watch/providers for TV ID {series_id_val}: {pe}")

        # Data Enrichment Logic: explicitly query reviews if missing in cached data
        if 'reviews' not in data:
            try:
                reviews_url = f"https://api.themoviedb.org/3/tv/{series_id_val}/reviews?api_key={api_key}"
                reviews_resp = get_resilient_session().get(reviews_url, timeout=3.0)
                if reviews_resp.status_code == 200:
                    reviews_data = reviews_resp.json()
                    data['reviews'] = reviews_data
                    # Write-through/enrich the cached payload in db & django cache
                    from django.core.cache import cache
                    cache.set(f"tv_detail_data_{series_id}", data, 86400)
                    CachedMedia.objects.update_or_create(
                        media_id=series_id_val,
                        media_type='tv',
                        defaults={'data': data}
                    )
            except Exception as re:
                print(f"[DATA ENRICHMENT WARNING] Failed to enrich reviews for TV show ID {series_id_val}: {re}")

        try:
            imdb_id = data.get('external_ids', {}).get('imdb_id')
            if imdb_id:
                from core.utils import fetch_omdb_data
                omdb_data = fetch_omdb_data(imdb_id)

            poster_path = data.get('poster_path') or ''
            backdrop_path = data.get('backdrop_path') or ''

            tv_show = {
                'id':                 data.get('id', series_id),
                'imdb_id':            imdb_id,
                'title':              data.get('name', 'Unknown Title'),
                'overview':           data.get('overview', ''),
                'first_air_date':     data.get('first_air_date', ''),
                'number_of_seasons':  data.get('number_of_seasons', 0),
                'number_of_episodes': data.get('number_of_episodes', 0),
                'vote_average':       round(data.get('vote_average', 0.0), 1),
                'genres':             [g.get('name', '') for g in data.get('genres', [])],
                'tagline':            data.get('tagline', ''),
                'production_companies': [
                    {
                        'name': c.get('name'),
                        'logo_url': f"https://image.tmdb.org/t/p/w92{c.get('logo_path')}" if c.get('logo_path') else None
                    }
                    for c in data.get('production_companies', []) if c.get('name')
                ][:4],
                'languages':            [l.get('english_name') for l in data.get('spoken_languages', []) if l.get('english_name')],
                'poster_url': (
                    f"https://image.tmdb.org/t/p/w500{poster_path}"
                    if poster_path else
                    'https://images.unsplash.com/photo-1593305841991-05c297ba4575?q=80&w=400&auto=format&fit=crop'
                ),
                'backdrop_url': (
                    f"https://image.tmdb.org/t/p/original{backdrop_path}"
                    if backdrop_path else ''
                ),
            }

            credits_payload = data.get('credits', {})
            raw_cast = credits_payload.get('cast', [])
            for member in raw_cast[:6]:
                profile_path = member.get('profile_path') or ''
                cast.append({
                    'id':         member.get('id'),
                    'name':       member.get('name', ''),
                    'character':  member.get('character', ''),
                    'profile_url': (
                        f"https://image.tmdb.org/t/p/w185{profile_path}"
                        if profile_path else
                        'https://ui-avatars.com/api/?name=' + urllib.parse.quote_plus(member.get('name', 'Actor'))
                    ),
                })

            videos_payload = data.get('videos', {})
            raw_videos = videos_payload.get('results', [])
            print(f"[DEBUG] tv_detail_view raw videos response count: {len(raw_videos)}")
            for idx, video in enumerate(raw_videos):
                print(f"[DEBUG] Video #{idx}: name={video.get('name')}, site={video.get('site')}, type={video.get('type')}, key={video.get('key')}, official={video.get('official')}")
            for video in raw_videos:
                if (
                    video.get('site') == 'YouTube'
                    and video.get('type') == 'Trailer'
                    and video.get('official', False)
                ):
                    trailer_key = video.get('key')
                    break
            if not trailer_key:
                for video in raw_videos:
                    if video.get('site') == 'YouTube' and video.get('type') == 'Trailer':
                        trailer_key = video.get('key')
                        break
            print(f"[DEBUG] tv_detail_view extracted trailer_key: {trailer_key}")

            providers_payload = data.get('watch/providers', {}).get('results', {})
            region_data = providers_payload.get('IN') or providers_payload.get('US')
            if not region_data:
                # Fallback to the first region with flatrate subscription streaming data
                for r_code, r_data in providers_payload.items():
                    if isinstance(r_data, dict) and 'flatrate' in r_data:
                        region_data = r_data
                        break
            if not region_data:
                region_data = {}
                
            raw_providers = region_data.get('flatrate', [])
            for p in raw_providers:
                logo_path = p.get('logo_path') or ''
                watch_providers.append({
                    'name':     p.get('provider_name', ''),
                    'logo_url': (
                        f"https://image.tmdb.org/t/p/w92{logo_path}"
                        if logo_path else ''
                    ),
                })

            # --- TasteDive TV Show Recommendations ---
            tastedive_success = False
            tv_title = tv_show.get('title')
            
            if tv_title:
                tastedive_key = getattr(settings, 'TASTEDIVE_API_KEY', '')
                tastedive_url = "https://tastedive.com/api/similar"
                td_params = {
                    'q': tv_title,
                    'type': 'show',
                    'info': 1,
                    'k': tastedive_key
                }
                try:
                    td_resp = get_resilient_session().get(tastedive_url, params=td_params, timeout=5.0)
                    if td_resp.status_code == 200:
                        td_data = td_resp.json()
                        raw_results = td_data.get('Similar', {}).get('Results', [])
                        for item in raw_results[:5]:
                            tastedive_recs.append({
                                'title': item.get('name', 'Unknown Title'),
                                'teaser': item.get('description', ''),
                                'wiki_url': item.get('wUrl', ''),
                                'youtube_url': item.get('yUrl', ''),
                                'youtube_id': item.get('yID', ''),
                            })
                        tastedive_success = True
                except Exception as te:
                    print(f"[TASTEDIVE ERROR] Failed to fetch TasteDive TV recommendations for '{tv_title}': {te}")

            if tastedive_success and tastedive_recs:
                tastedive_active = True
            else:
                tastedive_active = False
                # Fallback: Extract similar shows from TMDB payload
                similar_payload = data.get('similar', {})
                raw_similar = similar_payload.get('results', [])
                
                hidden_ids = []
                if request.user.is_authenticated:
                    hidden_ids = list(RecommendationFeedback.objects.filter(user=request.user, media_type='tv').values_list('media_id', flat=True))

                for s in raw_similar:
                    s_id = s.get('id')
                    if s_id in hidden_ids:
                        continue
                    s_poster = s.get('poster_path') or ''
                    similar_shows.append({
                        'id':           s_id,
                        'title':        s.get('name', 'Unknown'),
                        'vote_average': round(s.get('vote_average', 0.0), 1),
                        'poster_url': (
                            f"https://image.tmdb.org/t/p/w300{s_poster}"
                            if s_poster else
                            'https://images.unsplash.com/photo-1593305841991-05c297ba4575?q=80&w=400&auto=format&fit=crop'
                        ),
                    })
                    if len(similar_shows) >= 5:
                        break

            # Parse TMDB TV user reviews
            raw_tmdb_reviews = data.get('reviews', {}).get('results', [])
            for r in raw_tmdb_reviews:
                author_details = r.get('author_details', {})
                avatar_path = author_details.get('avatar_path') or ''
                if avatar_path.startswith('/http') or avatar_path.startswith('http'):
                    avatar_url = avatar_path[1:] if avatar_path.startswith('/') else avatar_path
                elif avatar_path:
                    avatar_url = f"https://image.tmdb.org/t/p/w92{avatar_path}"
                else:
                    avatar_url = f"https://ui-avatars.com/api/?name={urllib.parse.quote_plus(r.get('author', 'Critic'))}"
                
                rating = author_details.get('rating')
                
                tmdb_reviews.append({
                    'author': r.get('author', 'Anonymous'),
                    'avatar_url': avatar_url,
                    'rating': rating,
                    'content': r.get('content', ''),
                    'created_at': r.get('created_at', '')[:10],
                })
        except Exception as e:
            print(f"[TV DETAIL] Unexpected parsing error for series_id={series_id}: {e}")
            data = None

    if not data:
        # Fallback to local dict or placeholder
        local_title = "Content temporarily unavailable"
        if TV_DICT:
            try:
                df = pd.DataFrame(TV_DICT)
                match = df[df['id'] == series_id_val]
                if not match.empty:
                    local_title = match.iloc[0]['title']
            except Exception:
                pass
        tv_show = {
            'id': series_id_val,
            'title': local_title,
            'overview': 'We are currently experiencing connection timeout issues with the external API database. This content is temporarily unavailable. Please try again later.',
            'first_air_date': '',
            'number_of_seasons': 0,
            'number_of_episodes': 0,
            'vote_average': 0.0,
            'genres': ['Temporarily Offline'],
            'tagline': 'Offline Mode Enabled',
            'poster_url': 'https://images.unsplash.com/photo-1593305841991-05c297ba4575?q=80&w=400&auto=format&fit=crop',
            'backdrop_url': '',
        }

    is_in_watchlist = False
    if request.user.is_authenticated:
        is_in_watchlist = MovieWatchlist.objects.filter(
            user=request.user, media_id=series_id, media_type='tv'
        ).exists()

    # ── FETCH REVIEWS DATALAYER AND AUTHOR ACCESSORS ──
    raw_reviews = MediaReview.objects.filter(media_id=series_id, media_type='tv').select_related('user')
    reviews_list = []
    for r in raw_reviews:
        sentiment_label, sentiment_score = predict_sentiment(r.review_text)
        r.sentiment_label = sentiment_label
        r.sentiment_score = sentiment_score
        reviews_list.append(r)
        
    user_review = None
    if request.user.is_authenticated:
        for r in reviews_list:
            if r.user == request.user:
                user_review = r
                break

    # Fetch streaming links from Local DB / Cache (Write-Through batch retrieval)
    streaming_links = None
    if data:
        streaming_links = data.get('streaming_links')

    if streaming_links is None:
        from core.utils import get_streaming_links
        tv_title = tv_show.get('title', '')
        streaming_links = get_streaming_links(tv_title)
        if data:
            data['streaming_links'] = streaming_links
            from core.models import CachedMedia
            CachedMedia.objects.filter(media_id=series_id_val, media_type='tv').update(data=data)

    # Map streaming links directly to the watch_providers items with TMDB fallback
    from core.utils import normalize_name
    if watch_providers:
        for provider in watch_providers:
            provider_name = provider.get('name', '')
            norm_provider = normalize_name(provider_name)

            # Find matching link from Watchmode
            matched_url = None
            if streaming_links:
                for service_name, url in streaming_links.items():
                    norm_service = normalize_name(service_name)
                    if norm_provider == norm_service or norm_provider in norm_service or norm_service in norm_provider:
                        matched_url = url
                        break
            
            # Fallback to TMDB watch URL if no Watchmode link matches
            if not matched_url:
                matched_url = f"https://www.themoviedb.org/tv/{series_id}/watch?locale=IN"
                
            provider['web_url'] = matched_url

    # Data Debugging: Log the raw where_to_watch data to the terminal
    print(f"[DEBUG] tv_detail_view where_to_watch: {watch_providers}")

    # ── SOURCE FALLBACK STORYLINE RECONCILIATION ──
    tmdb_overview = tv_show.get('overview', '') if tv_show else ''
    omdb_plot = omdb_data.get('full_plot', '') if omdb_data else ''

    if omdb_plot and omdb_plot != 'N/A':
        storyline = omdb_plot
    else:
        storyline = tmdb_overview

    # ── SEASONS & EPISODES DATA ──
    seasons_list = data.get('seasons', []) if data else []
    seasons = [
        {
            'season_number': s.get('season_number'),
            'name': s.get('name', f"Season {s.get('season_number')}"),
            'episode_count': s.get('episode_count', 0)
        }
        for s in seasons_list if s.get('season_number') is not None and s.get('season_number') > 0
    ]
    
    # Sort seasons by season_number ascending
    seasons.sort(key=lambda x: x['season_number'])
    
    # Default to first season in the list, or Season 1
    active_season_number = 1
    if seasons:
        active_season_number = seasons[0]['season_number']
        
    episodes = []
    from django.core.cache import cache
    season_cache_key = f"tv_season_data_{series_id_val}_{active_season_number}"
    season_data = cache.get(season_cache_key)
    
    if not season_data:
        try:
            season_url = f"https://api.themoviedb.org/3/tv/{series_id_val}/season/{active_season_number}?api_key={api_key}&language=en-US"
            season_resp = get_resilient_session().get(season_url, timeout=5.0)
            if season_resp.status_code == 200:
                season_data = season_resp.json()
                cache.set(season_cache_key, season_data, 86400)
        except Exception as e:
            print(f"[TV DETAIL] Pre-fetching Season {active_season_number} episodes failed: {e}")
            season_data = None
            
    if season_data:
        raw_episodes = season_data.get('episodes', [])
        for ep in raw_episodes:
            still_path = ep.get('still_path') or ''
            episodes.append({
                'episode_number': ep.get('episode_number'),
                'name': ep.get('name', ''),
                'overview': ep.get('overview', ''),
                'runtime': ep.get('runtime', 0) or 0,
                'vote_average': round(ep.get('vote_average', 0.0), 1),
                'still_url': f"https://image.tmdb.org/t/p/original{still_path}" if still_path else 'https://images.unsplash.com/photo-1593305841991-05c297ba4575?q=80&w=400&auto=format&fit=crop'
            })

    context = {
        'tv_show':         tv_show,
        'tagline':         tv_show.get('tagline', ''),
        'storyline':       storyline,
        'cast':            cast,
        'trailer_key':     trailer_key,
        'watch_providers': watch_providers,
        'streaming_providers': watch_providers,
        'similar_shows':   similar_shows,
        'tastedive_recs':   tastedive_recs,
        'tastedive_active': tastedive_active,
        'is_in_watchlist': is_in_watchlist,
        'seasons':         seasons,
        'episodes':        episodes,
        # 💎 INJECTED REVIEWS DATA CONTEXTS
        'reviews':         reviews_list,
        'tmdb_reviews':    tmdb_reviews,
        'user_review':     user_review,
        'omdb_data':       omdb_data,
    }

    return render(request, 'core/tv_detail.html', context)


def tv_season_ajax(request, series_id, season_number):
    api_key = settings.TMDB_API_KEY
    from django.core.cache import cache
    from django.http import JsonResponse
    
    series_id_val = int(series_id)
    season_number_val = int(season_number)
    
    cache_key = f"tv_season_data_{series_id_val}_{season_number_val}"
    season_data = cache.get(cache_key)
    
    if not season_data:
        try:
            url = f"https://api.themoviedb.org/3/tv/{series_id_val}/season/{season_number_val}?api_key={api_key}&language=en-US"
            resp = get_resilient_session().get(url, timeout=5.0)
            if resp.status_code == 200:
                season_data = resp.json()
                cache.set(cache_key, season_data, 86400)
            else:
                return JsonResponse({'success': False, 'error': f"TMDB status code: {resp.status_code}"})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
            
    raw_episodes = season_data.get('episodes', []) if season_data else []
    episodes = []
    for ep in raw_episodes:
        still_path = ep.get('still_path') or ''
        episodes.append({
            'episode_number': ep.get('episode_number'),
            'name': ep.get('name', ''),
            'overview': ep.get('overview', ''),
            'runtime': ep.get('runtime', 0) or 0,
            'vote_average': round(ep.get('vote_average', 0.0), 1),
            'still_url': f"https://image.tmdb.org/t/p/original{still_path}" if still_path else 'https://images.unsplash.com/photo-1593305841991-05c297ba4575?q=80&w=400&auto=format&fit=crop'
        })
        
    return JsonResponse({'success': True, 'episodes': episodes})


# ======================================================================
# Watchlist Hub Dashboard View
# ======================================================================
def watchlist_hub_view(request):
    client = TMDBClient()
    watchlist_movies = []
    watchlist_tv = []
    user_reviews = []
    user_groups = []

    if request.user.is_authenticated:
        db_items = MovieWatchlist.objects.filter(user=request.user).order_by("-id")
        for item in db_items:
            cache_key = f"{item.media_type}_{item.media_id}"
            poster_url = POSTER_CACHE.get(cache_key)
            
            if not poster_url:
                poster_url = get_cached_poster(client, item.media_id, item.media_type)

            showcase_item = {
                'id': item.media_id,
                'poster_url': poster_url or 'https://images.unsplash.com/photo-1542204172-e7052809f852?q=80&w=400&auto=format&fit=crop',
            }

            if item.media_type == 'movie':
                watchlist_movies.append(showcase_item)
            else:
                watchlist_tv.append(showcase_item)

        user_reviews = Review.objects.filter(user=request.user).order_by('-created_at')
        from core.models import WatchGroup
        user_groups = WatchGroup.objects.filter(members=request.user).distinct()

    # ── QUERY SESSION BASED RECENTLY VIEWED ──
    recently_viewed_ids = request.session.get('recently_viewed', [])
    recently_viewed_media = []
    for item in recently_viewed_ids:
        # Backward compatibility if session contains old format integers
        if not isinstance(item, dict):
            m_id = item
            m_type = 'movie'
        else:
            m_id = item.get('id')
            m_type = item.get('type')
            
        if not m_id or not m_type:
            continue
            
        cache_key = f"{m_type}_{m_id}"
        poster_url = POSTER_CACHE.get(cache_key)
        if not poster_url:
            poster_url = get_cached_poster(client, m_id, m_type)
            
        recently_viewed_media.append({
            'id': m_id,
            'type': m_type,
            'poster_url': poster_url or ('https://images.unsplash.com/photo-1542204172-e7052809f852?q=80&w=400&auto=format&fit=crop' if m_type == 'movie' else 'https://images.unsplash.com/photo-1593305841991-05c297ba4575?q=80&w=400&auto=format&fit=crop'),
        })

    context = {
        'watchlist_movies':       watchlist_movies,
        'watchlist_tv':           watchlist_tv,
        'user_reviews':           user_reviews,
        'recently_viewed_media':  recently_viewed_media,
        'groups':                 user_groups,
    }
    return render(request, 'core/watchlist_hub.html', context)


# ======================================================================
# Interactive Media Reviews CRUD Controllers
# ======================================================================
@login_required
@require_POST
def add_media_review(request, media_type, media_id):
    review_text = request.POST.get('review_text', '').strip()
    
    # 1. Define these BEFORE you use them
    redirect_view_name = 'movie_detail' if media_type == 'movie' else 'tv_show_detail'
    redirect_param = 'movie_id' if media_type == 'movie' else 'series_id'
    
    # 2. Now the return statement will work
    if not review_text:
        return redirect(redirect_view_name, **{redirect_param: media_id})
    
    try:
        MediaReview.objects.update_or_create(
            user=request.user,
            media_id=int(media_id),
            media_type=media_type,
            defaults={'review_text': review_text}
        )
    except Exception as e:
        print(f"[REVIEW ENGINE] Error saving user review entry: {e}")
        
    return redirect(redirect_view_name, **{redirect_param: media_id})

@login_required
@require_POST
def update_media_review(request, review_id):
    review = get_object_or_404(Review, id=review_id)
    if review.user != request.user:
        return JsonResponse({'success': False, 'error': 'Unauthorized transaction request.'}, status=403)
    updated_text = request.POST.get('review_text', '').strip()
    if updated_text:
        review.content = updated_text
        review.save()
        return JsonResponse({'success': True, 'message': 'Review updated successfully.'})
    return JsonResponse({'success': False, 'error': 'Review text cannot be blank.'}, status=400)

@login_required
@require_POST
def delete_media_review(request, review_id):
    """
    Syllabus Reference: Unit 9.2 (CRUD - Delete with Authorization Guard)
    """
    review = get_object_or_404(Review, id=review_id)
    if review.user != request.user:
        return JsonResponse({'success': False, 'error': 'Unauthorized transaction request.'}, status=403)
    review.delete()
    return JsonResponse({'success': True, 'message': 'Review successfully scrubbed from catalog.'})


def get_provider_recommendations(provider_id, media_type='movie'):
    """
    Fetches the top 5 trending titles for a specific provider in India,
    including a direct deep-link to the TMDB watch page.
    """
    client = TMDBClient()
    # Dynamic URL based on media_type (movie/tv)
    url = f"{client.base_url}/discover/{media_type}"
    
    params = {
        'api_key': settings.TMDB_API_KEY,
        'with_watch_providers': provider_id,
        'watch_region': 'IN',
        'sort_by': 'popularity.desc',
        'language': 'en-US',
        'watch_monetization_types': 'flatrate'
    }
    
    import time
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            response = get_resilient_session().get(url, params=params, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])

                # Process results for the template
                processed_results = []
                for item in results[:10]:
                    item['media_type'] = media_type
                    
                    # Fetch cached poster
                    item['poster_url'] = get_cached_poster(client, item['id'], media_type)
                    
                    # Generate the direct watch link for the Indian locale
                    item['watch_url'] = f"https://www.themoviedb.org/{media_type}/{item['id']}/watch?locale=IN"
                    
                    processed_results.append(item)
                    
                return processed_results
            else:
                print(f"[PROVIDER FEED] Attempt {attempt+1}: API returned status code {response.status_code} for provider {provider_id}")
        except Exception as e:
            print(f"[PROVIDER FEED] Attempt {attempt+1}: Error fetching for provider {provider_id}: {e}")
            
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
            retry_delay *= 2
            
    return []


def person_profile(request, person_id):
    """
    Fetches the profile info and combined filmography (acting & directing) of a person using TMDB.
    """
    client = TMDBClient()
    
    # 1. Fetch person info (biography, birthday, place of birth, etc.)
    person_url = f"{client.base_url}/person/{person_id}?language=en-US"
    name = "Unknown Person"
    profile_path = None
    biography = ""
    birthday = None
    place_of_birth = None
    
    try:
        response = get_resilient_session().get(person_url, headers=client.headers, timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            name = data.get('name', 'Unknown Person')
            profile_path = data.get('profile_path')
            biography = data.get('biography', '')
            birthday = data.get('birthday')
            place_of_birth = data.get('place_of_birth')
    except Exception as e:
        print(f"[PERSON PROFILE] Error fetching details for {person_id}: {e}")
        
    # 2. Fetch filmography credits (Acting & Directing)
    credits_url = f"{client.base_url}/person/{person_id}/movie_credits?language=en-US"
    movies_map = {}
    
    try:
        response = get_resilient_session().get(credits_url, headers=client.headers, timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            
            # A. Process Cast Credits (Acting)
            for credit in data.get('cast', []):
                movie_id = credit.get('id')
                if not movie_id:
                    continue
                
                release_date = credit.get('release_date', '')
                year = release_date.split('-')[0] if release_date else 'N/A'
                character = credit.get('character', '').strip()
                role_str = f"As {character}" if character else "Actor"
                
                if movie_id not in movies_map:
                    poster_path = credit.get('poster_path')
                    movies_map[movie_id] = {
                        'movie_id': movie_id,
                        'title': credit.get('title') or credit.get('original_title') or 'Unknown Title',
                        'poster_url': f"{client.image_base_url}{poster_path}" if poster_path else client.movie_fallback,
                        'year': year,
                        'roles': [role_str]
                    }
                else:
                    movies_map[movie_id]['roles'].append(role_str)
                    
            # B. Process Crew Credits (Filtering for Directing / Director)
            for credit in data.get('crew', []):
                if credit.get('job') == 'Director':
                    movie_id = credit.get('id')
                    if not movie_id:
                        continue
                    
                    release_date = credit.get('release_date', '')
                    year = release_date.split('-')[0] if release_date else 'N/A'
                    role_str = "Director"
                    
                    if movie_id not in movies_map:
                        poster_path = credit.get('poster_path')
                        movies_map[movie_id] = {
                            'movie_id': movie_id,
                            'title': credit.get('title') or credit.get('original_title') or 'Unknown Title',
                            'poster_url': f"{client.image_base_url}{poster_path}" if poster_path else client.movie_fallback,
                            'year': year,
                            'roles': [role_str]
                          }
                    else:
                        if role_str not in movies_map[movie_id]['roles']:
                            movies_map[movie_id]['roles'].append(role_str)
                            
    except Exception as e:
        print(f"[PERSON PROFILE] Error fetching credits for {person_id}: {e}")
        
    # Convert dict to sorted list of movie records
    movies_list = list(movies_map.values())
    movies_list.sort(key=lambda x: x['year'], reverse=True)
    
    # Flatten list of roles to display nicely in template (e.g. "Director, As Sherlock Holmes")
    for m in movies_list:
        m['roles_display'] = ", ".join(m['roles'])
        
    profile_url = f"https://image.tmdb.org/t/p/w300{profile_path}" if profile_path else f"https://ui-avatars.com/api/?name={urllib.parse.quote_plus(name)}&background=2d1b4e&color=c084fc&size=300"
    
    # ── PAGINATION SYSTEM (15 ITEMS PER PAGE) ──
    paginator = Paginator(movies_list, 18)
    page_number = request.GET.get('page', 1)
    
    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)
        
    context = {
        'person_id': person_id,
        'name': name,
        'profile_url': profile_url,
        'biography': biography,
        'birthday': birthday,
        'place_of_birth': place_of_birth,
        'page_obj': page_obj
    }
    
    return render(request, 'core/person_profile.html', context)


@login_required
@require_POST
def submit_review(request, movie_id):
    """
    Syllabus Reference: Unit 9.2 WATCHLIST CRUDS (Create and Update reviews)
    Saves or updates a custom review for a movie.
    """
    movie_title = request.POST.get('movie_title', 'Unknown Movie').strip()
    rating_val = request.POST.get('rating')
    content = request.POST.get('content', '').strip()
    
    if not rating_val or not content:
        return redirect('movie_detail', movie_id=movie_id)
        
    try:
        rating = int(rating_val)
        if 1 <= rating <= 10:
            Review.objects.update_or_create(
                user=request.user,
                movie_id=movie_id,
                defaults={
                    'movie_title': movie_title,
                    'rating': rating,
                    'content': content
                }
            )
    except Exception as e:
        print(f"[REVIEW SAVE ERROR] {e}")
        
    return redirect('movie_detail', movie_id=movie_id)


def universal_search(request):
    """
    Query the TMDB /search/multi API using client headers.
    Groups results into 'Movies', 'TV Shows', and 'People'.
    """
    from django.http import JsonResponse
    import urllib.parse
    from .tmdb_api import TMDBClient
    
    query = request.GET.get('q', '').strip()
    if not query:
        return JsonResponse({'movies': [], 'tv_shows': [], 'people': []})

    client = TMDBClient()
    url = f"{client.base_url}/search/multi"
    params = {
        'query': query,
        'language': 'en-US',
        'page': 1,
        'include_adult': 'false'
    }
    
    try:
        response = get_resilient_session().get(url, headers=client.headers, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        results = data.get('results', [])
        
        movies = []
        tv_shows = []
        people = []
        
        for item in results:
            media_type = item.get('media_type')
            if media_type == 'movie':
                poster_path = item.get('poster_path')
                movies.append({
                    'id': item.get('id'),
                    'title': item.get('title') or item.get('original_title') or 'Unknown Movie',
                    'release_date': item.get('release_date', 'N/A'),
                    'poster': f"https://image.tmdb.org/t/p/w200{poster_path}" if poster_path else client.movie_fallback,
                    'rating': item.get('vote_average', 0.0)
                })
            elif media_type == 'tv':
                poster_path = item.get('poster_path')
                tv_shows.append({
                    'id': item.get('id'),
                    'name': item.get('name') or item.get('original_name') or 'Unknown TV Show',
                    'first_air_date': item.get('first_air_date', 'N/A'),
                    'poster': f"https://image.tmdb.org/t/p/w200{poster_path}" if poster_path else client.tv_fallback,
                    'rating': item.get('vote_average', 0.0)
                })
            elif media_type == 'person':
                profile_path = item.get('profile_path')
                known_for = [work.get('title') or work.get('name') for work in item.get('known_for', []) if work.get('title') or work.get('name')]
                people.append({
                    'id': item.get('id'),
                    'name': item.get('name') or 'Unknown Person',
                    'profile': f"https://image.tmdb.org/t/p/w200{profile_path}" if profile_path else f"https://ui-avatars.com/api/?name={urllib.parse.quote_plus(item.get('name', 'Actor'))}",
                    'known_for': ", ".join(known_for[:2])
                })
        
        return JsonResponse({
            'movies': movies[:5],
            'tv_shows': tv_shows[:5],
            'people': people[:5]
        })
    except Exception as e:
        print(f"[UNIVERSAL SEARCH] Error: {e}")
        return JsonResponse({'error': str(e)}, status=500)


def search_results_view(request):
    """
    Renders the dedicated search results page using TMDB search/multi query or discover query.
    Supports smart filters: genre, year, minimum rating, type, and sort.
    """
    from .tmdb_api import TMDBClient
    import urllib.parse
    import requests
    
    query = request.GET.get('q', '').strip()
    genre = request.GET.get('genre', '').strip()
    year = request.GET.get('year', '').strip()
    min_rating = request.GET.get('min_rating', '').strip()
    media_type = request.GET.get('type', 'all').strip()
    sort_by = request.GET.get('sort_by', 'popularity.desc').strip()

    client = TMDBClient()
    api_key = settings.TMDB_API_KEY
    
    movies = []
    tv_shows = []
    people = []

    def matches_filters(item, is_movie=True):
        if genre:
            genre_ids = item.get('genre_ids', [])
            if int(genre) not in genre_ids:
                return False
        if year:
            date_str = item.get('release_date') if is_movie else item.get('first_air_date')
            if not date_str or not date_str.startswith(year):
                return False
        if min_rating:
            try:
                if float(item.get('vote_average', 0.0)) < float(min_rating):
                    return False
            except ValueError:
                pass
        return True

    if query:
        # We query multi-search or separate search endpoints based on media_type
        # To get more candidate items to filter, we query pages 1 and 2
        for page in (1, 2):
            if media_type == 'movie':
                url = f"{client.base_url}/search/movie"
            elif media_type == 'tv':
                url = f"{client.base_url}/search/tv"
            else:
                url = f"{client.base_url}/search/multi"
                
            params = {
                'api_key': api_key,
                'query': query,
                'language': 'en-US',
                'page': page,
                'include_adult': 'false'
            }
            try:
                response = get_resilient_session().get(url, params=params, timeout=10.0)
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    for item in results:
                        item_type = item.get('media_type', media_type)
                        if item_type == 'movie' and (media_type == 'all' or media_type == 'movie'):
                            if matches_filters(item, is_movie=True):
                                poster_path = item.get('poster_path')
                                movies.append({
                                    'id': item.get('id'),
                                    'title': item.get('title') or item.get('original_title') or 'Unknown Movie',
                                    'release_date': item.get('release_date', 'N/A'),
                                    'poster_url': f"https://image.tmdb.org/t/p/w300{poster_path}" if poster_path else client.movie_fallback,
                                    'vote_average': round(item.get('vote_average', 0.0), 1),
                                    'overview': item.get('overview', '')
                                })
                        elif item_type == 'tv' and (media_type == 'all' or media_type == 'tv'):
                            if matches_filters(item, is_movie=False):
                                poster_path = item.get('poster_path')
                                tv_shows.append({
                                    'id': item.get('id'),
                                    'title': item.get('name') or item.get('original_name') or 'Unknown TV Show',
                                    'first_air_date': item.get('first_air_date', 'N/A'),
                                    'poster_url': f"https://image.tmdb.org/t/p/w300{poster_path}" if poster_path else client.tv_fallback,
                                    'vote_average': round(item.get('vote_average', 0.0), 1),
                                    'overview': item.get('overview', '')
                                })
                        elif item_type == 'person' and media_type == 'all':
                            profile_path = item.get('profile_path')
                            known_for = [work.get('title') or work.get('name') for work in item.get('known_for', []) if work.get('title') or work.get('name')]
                            people.append({
                                'id': item.get('id'),
                                'name': item.get('name') or 'Unknown Person',
                                'profile_url': f"https://image.tmdb.org/t/p/w300{profile_path}" if profile_path else f"https://ui-avatars.com/api/?name={urllib.parse.quote_plus(item.get('name', 'Actor'))}",
                                'known_for': ", ".join(known_for[:3])
                            })
            except Exception as e:
                print(f"[SEARCH VIEW] Error in page {page} query: {e}")
                
        # Remove duplicate records by ID
        seen_movies = set()
        movies = [m for m in movies if not (m['id'] in seen_movies or seen_movies.add(m['id']))]
        seen_tv = set()
        tv_shows = [t for t in tv_shows if not (t['id'] in seen_tv or seen_tv.add(t['id']))]

    elif genre or year or min_rating:
        # Discover Mode (filters active but no search text query)
        if media_type == 'all' or media_type == 'movie':
            url = f"{client.base_url}/discover/movie"
            params = {
                'api_key': api_key,
                'language': 'en-US',
                'sort_by': sort_by,
                'include_adult': 'false',
                'page': 1
            }
            if genre:
                params['with_genres'] = genre
            if year:
                params['primary_release_year'] = year
            if min_rating:
                params['vote_average.gte'] = min_rating
            try:
                response = get_resilient_session().get(url, params=params, timeout=10.0)
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    for item in results:
                        poster_path = item.get('poster_path')
                        movies.append({
                            'id': item.get('id'),
                            'title': item.get('title') or item.get('original_title') or 'Unknown Movie',
                            'release_date': item.get('release_date', 'N/A'),
                            'poster_url': f"https://image.tmdb.org/t/p/w300{poster_path}" if poster_path else client.movie_fallback,
                            'vote_average': round(item.get('vote_average', 0.0), 1),
                            'overview': item.get('overview', '')
                        })
            except Exception as e:
                print(f"[DISCOVER MOVIES] Error: {e}")

        if media_type == 'all' or media_type == 'tv':
            url = f"{client.base_url}/discover/tv"
            params = {
                'api_key': api_key,
                'language': 'en-US',
                'sort_by': sort_by,
                'include_adult': 'false',
                'page': 1
            }
            if genre:
                params['with_genres'] = genre
            if year:
                params['first_air_date_year'] = year
            if min_rating:
                params['vote_average.gte'] = min_rating
            try:
                response = get_resilient_session().get(url, params=params, timeout=10.0)
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    for item in results:
                        poster_path = item.get('poster_path')
                        tv_shows.append({
                            'id': item.get('id'),
                            'title': item.get('name') or item.get('original_name') or 'Unknown TV Show',
                            'first_air_date': item.get('first_air_date', 'N/A'),
                            'poster_url': f"https://image.tmdb.org/t/p/w300{poster_path}" if poster_path else client.tv_fallback,
                            'vote_average': round(item.get('vote_average', 0.0), 1),
                            'overview': item.get('overview', '')
                        })
            except Exception as e:
                print(f"[DISCOVER TV] Error: {e}")

    watchlist_movies = []
    watchlist_tv = []
    if request.user.is_authenticated:
        watchlist_movies = list(MovieWatchlist.objects.filter(
            user=request.user, media_type='movie'
        ).values_list('media_id', flat=True))
        watchlist_tv = list(MovieWatchlist.objects.filter(
            user=request.user, media_type='tv'
        ).values_list('media_id', flat=True))

    context = {
        'query':            query,
        'genre':            genre,
        'year':             year,
        'min_rating':       min_rating,
        'type':             media_type,
        'sort_by':          sort_by,
        'movies':           movies,
        'tv_shows':         tv_shows,
        'people':           people,
        'watchlist_movies': watchlist_movies,
        'watchlist_tv':     watchlist_tv,
    }
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return render(request, 'core/partials/search_results_grid.html', context)
        
    return render(request, 'core/search_results.html', context)


import random
from django.contrib import messages

def random_movie_view(request):
    """
    Selects a random movie from TMDB discover/movie (random page 1-500)
    and redirects the user directly to its detail view.
    """
    from django.conf import settings
    api_key = getattr(settings, 'TMDB_API_KEY', '')
    if not api_key:
        messages.error(request, 'Could not find a surprise, please try again!')
        return redirect('explore_movies')

    # Generate a random page number (TMDB allows up to 500 pages for discover)
    random_page = random.randint(1, 500)
    
    url = "https://api.themoviedb.org/3/discover/movie"
    params = {
        'api_key': api_key,
        'language': 'en-US',
        'sort_by': 'popularity.desc',
        'include_adult': 'false',
        'include_video': 'false',
        'page': random_page
    }
    
    try:
        response = get_resilient_session().get(url, params=params, timeout=10.0)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                random_movie = random.choice(results)
                movie_id = random_movie.get('id')
                if movie_id:
                    return redirect('movie_detail', movie_id=movie_id)
        
        # Fallback to page 1 if the randomly generated page fails
        params['page'] = 1
        response = get_resilient_session().get(url, params=params, timeout=10.0)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                random_movie = random.choice(results)
                movie_id = random_movie.get('id')
                if movie_id:
                    return redirect('movie_detail', movie_id=movie_id)
    except Exception as e:
        print(f"[RANDOM MOVIE ERROR] Failed to fetch random movie: {e}")

    messages.error(request, 'Could not find a surprise, please try again!')
    return redirect('explore_movies')


def random_tv_show_view(request):
    """
    Selects a random TV show from TMDB discover/tv (random page 1-500)
    and redirects the user directly to its detail view.
    """
    from django.conf import settings
    api_key = getattr(settings, 'TMDB_API_KEY', '')
    if not api_key:
        messages.error(request, 'Could not find a surprise, please try again!')
        return redirect('explore_tv')

    # Generate a random page number (TMDB allows up to 500 pages for discover)
    random_page = random.randint(1, 500)
    
    url = "https://api.themoviedb.org/3/discover/tv"
    params = {
        'api_key': api_key,
        'language': 'en-US',
        'sort_by': 'popularity.desc',
        'include_adult': 'false',
        'page': random_page
    }
    
    try:
        response = get_resilient_session().get(url, params=params, timeout=10.0)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                random_show = random.choice(results)
                series_id = random_show.get('id')
                if series_id:
                    return redirect('tv_show_detail', series_id=series_id)
        
        # Fallback to page 1 if the randomly generated page fails
        params['page'] = 1
        response = get_resilient_session().get(url, params=params, timeout=10.0)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                random_show = random.choice(results)
                series_id = random_show.get('id')
                if series_id:
                    return redirect('tv_show_detail', series_id=series_id)
    except Exception as e:
        print(f"[RANDOM TV ERROR] Failed to fetch random TV show: {e}")

    messages.error(request, 'Could not find a surprise, please try again!')
    return redirect('explore_tv')


def movies_by_genre_view(request, genre_name):
    """
    Renders the catalog list filtered by the chosen genre name.
    """
    from core.utils import fetch_media_by_genre
    from core.models import MovieWatchlist
    from django.shortcuts import render, redirect
    from django.contrib import messages

    page_number = request.GET.get('page', 1)
    try:
        page_number = int(page_number)
    except ValueError:
        page_number = 1

    # Genre-Level Result Caching: Query GenreCache first
    from core.models import GenreCache
    cache_name = f"movie_{genre_name.strip().lower()}_page_{page_number}"
    cached = GenreCache.objects.filter(genre_name=cache_name).first()
    
    if cached:
        data_payload = cached.data or {}
        movies_records = data_payload.get('records', [])
        target_genre_name = data_payload.get('target_genre_name', genre_name)
        total_pages = data_payload.get('total_pages', 1)
    else:
        try:
            movies_records, target_genre_name, total_pages = fetch_media_by_genre(genre_name, media_type="movie", page=page_number)
            if movies_records:
                payload = {
                    'records': movies_records,
                    'target_genre_name': target_genre_name,
                    'total_pages': total_pages
                }
                GenreCache.objects.update_or_create(
                    genre_name=cache_name,
                    defaults={'data': payload}
                )
        except Exception as e:
            print(f"[GENRE VIEW ERROR] Silent fallback triggered for movie: {e}")
            movies_records = []
            target_genre_name = genre_name
            total_pages = 1

    class MockPage:
        def __init__(self, number, object_list, max_pages):
            self.number = number
            self.object_list = object_list
            self.has_previous = number > 1
            self.previous_page_number = number - 1
            self.has_next = number < max_pages
            self.next_page_number = number + 1
            self.has_other_pages = max_pages > 1

    class MockPaginator:
        def __init__(self, max_pages):
            self.max_pages = max_pages

        @property
        def num_pages(self):
            return self.max_pages

    page_obj = MockPage(page_number, movies_records, total_pages)
    page_obj.paginator = MockPaginator(total_pages)

    watchlist_ids = []
    if request.user.is_authenticated:
        watchlist_ids = list(MovieWatchlist.objects.filter(
            user=request.user, media_type='movie'
        ).values_list('media_id', flat=True))

    context = {
        'movies':       movies_records,
        'page_obj':     page_obj,
        'watchlist_ids': watchlist_ids,
        'query':         "",
        'selected_genre': target_genre_name,
    }

    is_ajax = request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.GET.get('ajax') == 'true'
    if is_ajax:
        return render(request, 'core/includes/movie_grid_partial.html', context)
    return render(request, 'core/explore_movies.html', context)


def tv_shows_by_genre_view(request, genre_name):
    """
    Renders the catalog list of TV shows filtered by the chosen genre name.
    """
    from core.utils import fetch_media_by_genre
    from core.models import MovieWatchlist
    from django.shortcuts import render, redirect
    from django.contrib import messages

    page_number = request.GET.get('page', 1)
    try:
        page_number = int(page_number)
    except ValueError:
        page_number = 1

    # Genre-Level Result Caching: Query GenreCache first
    from core.models import GenreCache
    cache_name = f"tv_{genre_name.strip().lower()}_page_{page_number}"
    cached = GenreCache.objects.filter(genre_name=cache_name).first()
    
    if cached:
        data_payload = cached.data or {}
        tv_records = data_payload.get('records', [])
        target_genre_name = data_payload.get('target_genre_name', genre_name)
        total_pages = data_payload.get('total_pages', 1)
    else:
        try:
            tv_records, target_genre_name, total_pages = fetch_media_by_genre(genre_name, media_type="tv", page=page_number)
            if tv_records:
                payload = {
                    'records': tv_records,
                    'target_genre_name': target_genre_name,
                    'total_pages': total_pages
                }
                GenreCache.objects.update_or_create(
                    genre_name=cache_name,
                    defaults={'data': payload}
                )
        except Exception as e:
            print(f"[GENRE VIEW ERROR] Silent fallback triggered for tv: {e}")
            tv_records = []
            target_genre_name = genre_name
            total_pages = 1

    class MockPage:
        def __init__(self, number, object_list, max_pages):
            self.number = number
            self.object_list = object_list
            self.has_previous = number > 1
            self.previous_page_number = number - 1
            self.has_next = number < max_pages
            self.next_page_number = number + 1
            self.has_other_pages = max_pages > 1

    class MockPaginator:
        def __init__(self, max_pages):
            self.max_pages = max_pages

        @property
        def num_pages(self):
            return self.max_pages

    page_obj = MockPage(page_number, tv_records, total_pages)
    page_obj.paginator = MockPaginator(total_pages)

    watchlist_ids = []
    if request.user.is_authenticated:
        watchlist_ids = list(MovieWatchlist.objects.filter(
            user=request.user, media_type='tv'
        ).values_list('media_id', flat=True))

    context = {
        'tv_shows':      tv_records,
        'page_obj':      page_obj,
        'watchlist_ids': watchlist_ids,
        'query':         "",
        'selected_genre': target_genre_name,
    }

    is_ajax = request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.GET.get('ajax') == 'true'
    if is_ajax:
        return render(request, 'core/includes/tv_grid_partial.html', context)
    return render(request, 'core/explore_tv.html', context)


# ======================================================================
# Collaborative Watch Groups (Watch Party Feature)
# ======================================================================
from django.shortcuts import get_object_or_404
from django.core.exceptions import PermissionDenied
from django.contrib import messages
from django.http import Http404

@login_required
def group_list_view(request):
    """
    Renders groups_list.html showing all groups the user created or joined.
    """
    from core.models import WatchGroup
    user_groups = WatchGroup.objects.filter(members=request.user).distinct()
    return render(request, 'core/groups_list.html', {'groups': user_groups})


@login_required
@require_POST
def create_group_view(request):
    """
    POST handler to create a new Watch Group.
    """
    from core.models import WatchGroup
    name = request.POST.get('name', '').strip()
    if not name:
        messages.error(request, "Group name cannot be blank.")
        return redirect('group_list')

    # Create the watch group
    group = WatchGroup.objects.create(name=name, creator=request.user)
    # The creator is automatically a member
    group.members.add(request.user)
    
    messages.success(request, f"Watch Group '{group.name}' created successfully!")
    return redirect('group_dashboard', group_code=group.invite_code)


@login_required
def join_group_view(request, invite_code):
    """
    Resolves the UUID invite link and adds the logged-in user to the group's members list.
    """
    from core.models import WatchGroup
    group = get_object_or_404(WatchGroup, invite_code=invite_code)
    
    if request.user not in group.members.all():
        group.members.add(request.user)
        messages.success(request, f"You have successfully joined the Watch Group '{group.name}'!")
    else:
        messages.info(request, f"You are already a member of '{group.name}'.")
        
    return redirect('group_dashboard', group_code=group.invite_code)


@login_required
def group_dashboard_view(request, group_code):
    """
    Renders group_dashboard.html if request.user is a member of the group.
    """
    from core.models import WatchGroup, SharedWatchlist, GroupMessage
    group = get_object_or_404(WatchGroup, invite_code=group_code)
    
    # Assert membership security access controls
    if request.user not in group.members.all():
        raise PermissionDenied("You are not a member of this Watch Group.")

    shared_items = group.watchlist_items.select_related('added_by').all()
    chat_messages = group.messages.select_related('sender').all()
    
    # Build full absolute invite link for sharing
    from django.urls import reverse
    invite_url = request.build_absolute_uri(reverse('join_watch_group', args=[group.invite_code]))

    context = {
        'group': group,
        'shared_items': shared_items,
        'invite_url': invite_url,
        'chat_messages': chat_messages,
    }
    return render(request, 'core/group_dashboard.html', context)


@login_required
@require_POST
def send_group_message_view(request, group_code):
    """
    AJAX POST handler to save and broadcast a new group message.
    """
    from core.models import WatchGroup, GroupMessage
    group = get_object_or_404(WatchGroup, invite_code=group_code)
    
    # Assert security membership
    if request.user not in group.members.all():
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)
        
    content = request.POST.get('content', '').strip()
    if not content:
        return JsonResponse({'success': False, 'error': 'Message content cannot be blank.'}, status=400)
        
    msg = GroupMessage.objects.create(
        group=group,
        sender=request.user,
        content=content
    )
    
    # Bulk notify other members of the group
    try:
        from core.models import Notification
        from django.urls import reverse
        other_members = group.members.exclude(id=request.user.id)
        notifications = [
            Notification(
                recipient=member,
                sender=request.user,
                message=f"@{request.user.username} sent a message in '{group.name}': \"{content[:30]}...\"",
                target_url=reverse('group_dashboard', args=[group.invite_code])
            )
            for member in other_members
        ]
        Notification.objects.bulk_create(notifications)
    except Exception as ne:
        print(f"[NOTIFY ERROR] Failed to send chat notifications: {ne}")
    
    return JsonResponse({
        'success': True,
        'message': {
            'id': msg.id,
            'sender': msg.sender.username,
            'sender_initials': msg.sender.username[:2].upper(),
            'content': msg.content,
            'created_at': msg.created_at.strftime('%b %d, %H:%M')
        }
    })


@login_required
@require_POST
def add_group_item_view(request, group_code):
    """
    AJAX POST endpoint to dynamically append a title to the group's shared watchlist.
    """
    from core.models import WatchGroup, SharedWatchlist
    group = get_object_or_404(WatchGroup, invite_code=group_code)
    
    if request.user not in group.members.all():
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)
        
    media_id_raw = request.POST.get('media_id')
    media_type = request.POST.get('media_type', 'movie')
    title = request.POST.get('title', '').strip()
    poster_url = request.POST.get('poster_url', '').strip()
    
    if not media_id_raw or not title:
        return JsonResponse({'success': False, 'error': 'Missing required fields (media_id, title).'}, status=400)
        
    try:
        media_id = int(media_id_raw)
    except ValueError:
        return JsonResponse({'success': False, 'error': 'Invalid media_id.'}, status=400)

    # Prevent duplicates in the shared watchlist
    exists = SharedWatchlist.objects.filter(group=group, media_id=media_id, media_type=media_type).exists()
    if exists:
        return JsonResponse({'success': False, 'error': 'This title is already on the shared watchlist.'}, status=400)

    item = SharedWatchlist.objects.create(
        group=group,
        media_id=media_id,
        media_type=media_type,
        title=title,
        poster_url=poster_url,
        added_by=request.user
    )

    # Bulk notify other members of the group
    try:
        from core.models import Notification
        from django.urls import reverse
        other_members = group.members.exclude(id=request.user.id)
        notifications = [
            Notification(
                recipient=member,
                sender=request.user,
                message=f"@{request.user.username} added '{title}' to the watch group '{group.name}'.",
                target_url=reverse('group_dashboard', args=[group.invite_code])
            )
            for member in other_members
        ]
        Notification.objects.bulk_create(notifications)
    except Exception as ne:
        print(f"[NOTIFY ERROR] Failed to send watchlist item notifications: {ne}")

    return JsonResponse({
        'success': True,
        'message': f"'{title}' added to the shared watchlist!",
        'item': {
            'id': item.id,
            'title': item.title,
            'media_type': item.get_media_type_display(),
            'poster_url': item.poster_url,
            'added_by': item.added_by.username,
        }
    })


@login_required
@require_POST
def remove_group_item_view(request, group_code, item_id):
    """
    AJAX POST endpoint to securely delete an item from the Watch Group's shared watchlist.
    Verifies that the request user is a member of the group before executing.
    """
    from core.models import WatchGroup, SharedWatchlist
    group = get_object_or_404(WatchGroup, invite_code=group_code)
    
    # Verify group membership
    if request.user not in group.members.all():
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)
        
    item = get_object_or_404(SharedWatchlist, id=item_id, group=group)
    title = item.title
    item.delete()
    
    return JsonResponse({
        'success': True,
        'message': f"'{title}' removed from group watchlist."
    })


# ======================================================================
# In-App Notification System Views
# ======================================================================
@login_required
def notifications_list_view(request):
    """
    Renders the dedicated notifications list dashboard.
    """
    from core.models import Notification
    notifications = Notification.objects.filter(recipient=request.user)
    return render(request, 'core/notifications.html', {'notifications': notifications})


@login_required
def mark_notification_read(request, notification_id):
    """
    Marks a single notification as read and redirects the user to the target link destination.
    """
    from core.models import Notification
    notification = get_object_or_404(Notification, id=notification_id, recipient=request.user)
    notification.is_read = True
    notification.save()
    return redirect(notification.target_url)


@login_required
def mark_all_notifications_read(request):
    """
    Mark all unread notifications for the user as read.
    """
    from core.models import Notification
    Notification.objects.filter(recipient=request.user, is_read=False).update(is_read=True)
    messages.success(request, "All notifications marked as read.")
    return redirect('notifications_list')


BG_LOGS = []

@login_required
def temp_load_fixture(request):
    """
    Temporary route to load pre-computed recommendations fixture on the free Render instance.
    Runs asynchronously in a background thread to prevent 30s request gateway timeouts.
    """
    global BG_LOGS
    from core.models import CachedRecommendation
    from django.http import HttpResponse
    import os
    from django.conf import settings
    import threading

    count = CachedRecommendation.objects.count()
    fixture_path = os.path.join(settings.BASE_DIR, 'recommendations.json')
    exists = os.path.exists(fixture_path)
    size = os.path.getsize(fixture_path) if exists else 0

    if request.GET.get('trigger') == 'true':
        BG_LOGS = []  # Reset logs
        def load_fixture_bg():
            import json
            BG_LOGS.append("Starting background fast bulk-create import...")
            try:
                fixture_path_bg = os.path.join(settings.BASE_DIR, 'recommendations.json')
                BG_LOGS.append("Reading JSON file into memory...")
                with open(fixture_path_bg, 'r') as f:
                    data = json.load(f)
                
                BG_LOGS.append(f"Loaded {len(data)} records from JSON. Clearing old recommendations...")
                CachedRecommendation.objects.all().delete()
                
                BG_LOGS.append("Converting to model instances...")
                recs = []
                for item in data:
                    fields = item['fields']
                    recs.append(CachedRecommendation(
                        source_id=fields['source_id'],
                        target_id=fields['target_id'],
                        score=fields['score'],
                        media_type=fields['media_type']
                    ))
                
                BG_LOGS.append("Executing bulk create database inserts in chunks of 20,000...")
                chunk_size = 20000
                for i in range(0, len(recs), chunk_size):
                    chunk = recs[i:i+chunk_size]
                    CachedRecommendation.objects.bulk_create(chunk)
                    BG_LOGS.append(f"Inserted {i + len(chunk)} / {len(recs)} records...")
                
                BG_LOGS.append("Successfully loaded recommendations fixture using fast bulk create!")
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                BG_LOGS.append(f"Error loading fixture: {e}")
                BG_LOGS.append(error_trace.replace("\n", "<br>"))

        t = threading.Thread(target=load_fixture_bg)
        t.daemon = True
        t.start()
        return HttpResponse(f"Background loading task triggered. Refresh this page to view logs.")

    logs_str = "<br>".join(BG_LOGS)
    return HttpResponse(
        f"<b>Database Count:</b> {count}<br>"
        f"<b>Fixture File Path:</b> {fixture_path}<br>"
        f"<b>Fixture File Exists:</b> {exists} (Size: {size} bytes)<br><br>"
        f"<b>Logs from last run:</b><br>{logs_str}<br><br>"
        f"To trigger loading, append ?trigger=true to the URL."
    )


from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
from core.models import ContinueWatching

@csrf_exempt
def sync_continue_watching_view(request):
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'error': 'User not authenticated'}, status=401)
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            if isinstance(data, list):
                for item in data:
                    media_id = str(item.get('id', '')).strip()
                    media_type = item.get('type', '')
                    title = item.get('title', '')
                    poster_url = item.get('poster_url', '')
                    season = item.get('season')
                    episode = item.get('episode')
                    episode_title = item.get('episode_title')
                    total_episodes = item.get('total_episodes_in_season')
                    
                    if not media_id or media_type not in ['movie', 'tv']:
                        continue
                    
                    try:
                        season = int(season) if season is not None and str(season).isdigit() else None
                    except (ValueError, TypeError):
                        season = None
                        
                    try:
                        episode = int(episode) if episode is not None and str(episode).isdigit() else None
                    except (ValueError, TypeError):
                        episode = None

                    try:
                        total_episodes = int(total_episodes) if total_episodes is not None and str(total_episodes).isdigit() else None
                    except (ValueError, TypeError):
                        total_episodes = None
                    
                    ContinueWatching.objects.update_or_create(
                        user=request.user,
                        media_id=media_id,
                        media_type=media_type,
                        defaults={
                            'title': title,
                            'poster_url': poster_url,
                            'season': season,
                            'episode': episode,
                            'episode_title': episode_title,
                            'total_episodes_in_season': total_episodes
                        }
                    )
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=400)
            
    db_items = ContinueWatching.objects.filter(user=request.user).order_by('-last_watched')[:12]
    result = []
    for item in db_items:
        result.append({
            'id': item.media_id,
            'type': item.media_type,
            'title': item.title,
            'poster_url': item.poster_url,
            'season': item.season,
            'episode': item.episode,
            'episode_title': item.episode_title,
            'total_episodes_in_season': item.total_episodes_in_season,
            'last_watched': item.last_watched.isoformat()
        })
        
    return JsonResponse({'success': True, 'continue_watching': result})

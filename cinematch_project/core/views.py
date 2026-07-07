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
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_protect
from django.http import JsonResponse
from django.conf import settings
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.utils.html import escape

from .models import UserProfile, MovieWatchlist, MediaReview, Review
from .tmdb_api import TMDBClient

# ======================================================================
# Global Memory Cache System (Unit 9.2 & 7 Optimization)
# ======================================================================
# PERSISTENT RAM CACHE: Persists fetched poster URLs across requests.
POSTER_CACHE = {}

# Global variables to cache high-dimensional vector similarity matrices
MOVIE_DICT = None
MOVIE_SIMILARITY = None
TV_DICT = None
TV_SIMILARITY = None

# TMDB Genre ID to Name Mapping dictionary
TMDB_GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Sci-Fi",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
}

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
        poster_url = client.get_media_assets(media_id, media_type, timeout=5.0)
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
            movie_dict_path = os.path.join(settings.BASE_DIR, 'movie_dict.pkl')
            movie_sim_path = os.path.join(settings.BASE_DIR, 'similarity.pkl')
            tv_dict_path = os.path.join(settings.BASE_DIR, 'tv_dict.pkl')
            tv_sim_path = os.path.join(settings.BASE_DIR, 'tv_similarity.pkl')

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

def get_recommendations(user_watchlist_ids, media_type='movie'):
    """
    Syllabus Reference: Units 4 & 5 Model Inference and Metric Evaluation
    Mathematical Concept: Similarity Matrix Vector Aggregation
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
        return []
        
    df = pd.DataFrame(data_dict)
    
    # ── COLD START LOGIC & FALLBACK LAYER 1: Watchlist is completely empty ──
    if not user_watchlist_ids:
        if 'popularity' in df.columns:
            trending_df = df.sort_values(by='popularity', ascending=False)
        else:
            trending_df = df
        defaults = trending_df.head(8).to_dict(orient='records')
        for d in defaults:
            title_text = d.get('title') or d.get('name') or 'Unknown Title'
            d['title'] = title_text
            d['poster_url'] = get_cached_poster(client, d[id_col], media_type)
            d['watch_link'] = client.get_streaming_or_theatre_links(title_text, media_type, False)
        return defaults

    # Find row indices of user's saved titles inside the catalog DataFrame
    watchlist_indices = df[df[id_col].isin(user_watchlist_ids)].index.tolist()
    
    # ── COLD START LOGIC & FALLBACK LAYER 2: Watchlist IDs do not match dataset records ──
    if not watchlist_indices:
        if 'popularity' in df.columns:
            trending_df = df.sort_values(by='popularity', ascending=False)
        else:
            trending_df = df
        defaults = trending_df.head(8).to_dict(orient='records')
        for d in defaults:
            title_text = d.get('title') or d.get('name') or 'Unknown Title'
            d['title'] = title_text
            d['poster_url'] = get_cached_poster(client, d[id_col], media_type)
            d['watch_link'] = client.get_streaming_or_theatre_links(title_text, media_type, False)
        return defaults
        
    try:
        aggregated_sim = np.sum(sim_matrix[watchlist_indices], axis=0)

        if media_type == 'tv' and len(user_watchlist_ids) > 0:
            aggregated_sim = aggregated_sim / len(user_watchlist_ids)

        sorted_indices = np.argsort(aggregated_sim)[::-1]
        
        # ── HYBRID SEARCH STRATEGY ──
        # Apply a popularity filter to de-emphasize obscure similarities.
        # Only show recommendations that meet a minimum popularity threshold (e.g. 15.0).
        if media_type == 'tv' and 'popularity' in df.columns:
            popularity_threshold = 15.0
            recommended_indices = [
                idx for idx in sorted_indices 
                if idx not in watchlist_indices 
                and df.iloc[idx].get('popularity', 0) >= popularity_threshold
            ]
        else:
            recommended_indices = [idx for idx in sorted_indices if idx not in watchlist_indices]
            
        top_indices = recommended_indices[:8]
        recommendations = df.iloc[top_indices].to_dict(orient='records')
        
        for rec in recommendations:
            media_id = rec[id_col]
            title_text = rec.get('title') or rec.get('name') or 'Unknown Title'
            rec['title'] = title_text
            rec['poster_url'] = get_cached_poster(client, media_id, media_type)
            rec['watch_link'] = client.get_streaming_or_theatre_links(title_text, media_type, False)

        if media_type == 'tv':
            raw_similarity_scores = np.sort(aggregated_sim)[::-1]
            print(f"\n[TV ML DIAGNOSTIC] Target Watchlist Input IDs: {user_watchlist_ids}")
            print(f"[TV ML DIAGNOSTIC] Top 5 Matrix Match Scores: {[score for score in raw_similarity_scores[:5]]}")

        return recommendations
    except Exception as e:
        print(f"[ML INFERENCE] Error during recommendation synthesis: {e}")
        return []

# ======================================================================
# User Authentication, Registration, and Home Views
# ======================================================================
from django.contrib import messages
from django.contrib.auth import logout, authenticate
from django.contrib.auth.forms import AuthenticationForm

@csrf_protect
def signup_view(request):
    if request.user.is_authenticated:
        return redirect('for_you_feed')

    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            UserProfile.objects.get_or_create(user=user)
            login(request, user)
            messages.success(request, f"Welcome to CineMatch, {user.username}! Your account was created successfully.")
            return redirect('for_you_feed')
        else:
            messages.error(request, "Please correct the registration errors below.")
    else:
        form = UserCreationForm()
        
    return render(request, 'core/signup.html', {'form': form})

# Maintain register_user as a compatibility alias for signup_view
register_user = signup_view

@csrf_protect
def login_view(request):
    if request.user.is_authenticated:
        return redirect('for_you_feed')

    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
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
        form = AuthenticationForm()
        
    return render(request, 'core/login.html', {'form': form})

def logout_view(request):
    logout(request)
    messages.info(request, "You have been logged out successfully.")
    return redirect('login')

def home_view(request):
    if request.user.is_authenticated:
        return redirect('for_you_feed')
    if request.method == 'POST':
        return signup_view(request)
    return render(request, 'core/signup.html', {'form': UserCreationForm()})

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

        if deleted_count > 0:
            return JsonResponse({'success': True, 'message': 'Title successfully removed from watchlist.'})
        else:
            return JsonResponse({'success': False, 'error': 'Title not found in your watchlist.'}, status=404)
    except ValueError:
        return JsonResponse({'success': False, 'error': 'media_id must be a valid integer.'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

# ======================================================================
# Home / "For You" Feed Loader (Read & Recommend)
# ======================================================================
@login_required
def for_you_feed(request):
    user = request.user
    client = TMDBClient()
    
    watchlist_items = MovieWatchlist.objects.filter(user=user)
    saved_movies = list(watchlist_items.filter(media_type='movie').values_list('media_id', flat=True))
    saved_tv_shows = list(watchlist_items.filter(media_type='tv').values_list('media_id', flat=True))
    saved_ids = list(watchlist_items.values_list('media_id', flat=True))
    
    spotlight_movie = {
        'movie_id': 157336,
        'title': 'Interstellar',
        'overview': 'The adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances involved in an interstellar voyage.',
        'backdrop_url': 'https://image.tmdb.org/t/p/original/rAiYw1jKe6vS8v36ZasYwB66G6B.jpg'
    }
    
    now_showing = []
    url = f"{client.base_url}/movie/now_playing?language=en-US&region=IN&page=1"
    
    try:
        response = requests.get(url, headers=client.headers, timeout=5.0)
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
    
    recommended_movies = get_recommendations(saved_movies, 'movie')
    recommended_tv_shows = get_recommendations(saved_tv_shows, 'tv')

    # ── FETCH WEEKLY TRENDING MOVIES FROM TMDB ──
    weekly_trending = []
    try:
        trending_url = f"{client.base_url}/trending/movie/week?language=en-US"
        trending_resp = requests.get(trending_url, headers=client.headers, timeout=2.5)
        if trending_resp.status_code == 200:
            results = trending_resp.json().get('results', [])
            for item in results[:10]:
                release_date = item.get('release_date', '')
                year = release_date.split('-')[0] if release_date else 'N/A'
                weekly_trending.append({
                    'id': item.get('id'),
                    'title': item.get('title'),
                    'poster_url': get_cached_poster(client, item.get('id'), 'movie'),
                    'year': year,
                    'release_date': release_date
                })
    except Exception as e:
        print(f"[WEEKLY TRENDING] TMDB API request exception: {e}")

    if not weekly_trending:
        defaults_list = [
            {'id': 27205, 'title': 'Inception', 'year': '2010', 'release_date': '2010-07-16'},
            {'id': 157336, 'title': 'Interstellar', 'year': '2014', 'release_date': '2014-11-07'},
            {'id': 19995, 'title': 'Avatar', 'year': '2009', 'release_date': '2009-12-18'},
            {'id': 49026, 'title': 'The Dark Knight Rises', 'year': '2012', 'release_date': '2012-07-20'},
            {'id': 24428, 'title': 'The Avengers', 'year': '2012', 'release_date': '2012-05-04'}
        ]
        for item in defaults_list:
            weekly_trending.append({
                'id': item['id'],
                'title': item['title'],
                'poster_url': get_cached_poster(client, item['id'], 'movie'),
                'year': item['year'],
                'release_date': item['release_date']
            })

    talk_of_town = weekly_trending[:3]
    most_interested = weekly_trending[:5]
    
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
    
    # Hotstar (122)
    hotstar_movies = get_provider_recommendations(122, 'movie')[:5]
    hotstar_tv = get_provider_recommendations(122, 'tv')[:5]
    hotstar_data = hotstar_movies + hotstar_tv
    random.shuffle(hotstar_data)
    for film in hotstar_data:
        if 'title' not in film: film['title'] = film.get('name', 'N/A')
        if 'name' not in film: film['name'] = film.get('title', 'N/A')
    
    # ── DEBUG LOGGING FOR USER QUERY VERIFICATION ──
    print(f"[DEBUG] netflix_movies count: {len(netflix_data)}")
    if not netflix_data:
        print("[WARNING] netflix_movies query returned 0 rows or variable is empty!")
    else:
        print(f"[DEBUG] First entry of netflix_movies: {netflix_data[0]}")
        
    context = {
        'watchlist_count': watchlist_items.count(),
        'saved_movies': saved_movies,
        'saved_tv_shows': saved_tv_shows,
        'saved_ids': saved_ids,
        'now_showing': now_showing,
        'recommended_movies': recommended_movies,
        'recommended_tv_shows': recommended_tv_shows,
        'spotlight_movie': spotlight_movie,
        'talk_of_town': talk_of_town,
        'most_interested': most_interested,
        # 💎 INJECTED INTO CONTEXT
        'netflix_movies': netflix_data,
        'prime_movies': prime_data,
        'hotstar_movies': hotstar_data,
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
            # 💎 INJECTED INTO JSON
            'netflix_movies': netflix_data,
            'prime_movies': prime_data,
            'hotstar_movies': hotstar_data,
        })
        
    return render(request, 'core/for_you.html', context)
# ======================================================================
# Explore Movies View
# ======================================================================
@login_required
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

    watchlist_ids = list(MovieWatchlist.objects.filter(
        user=request.user, media_type='movie'
    ).values_list('media_id', flat=True))

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
@login_required
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

    watchlist_ids = list(MovieWatchlist.objects.filter(
        user=request.user, media_type='tv'
    ).values_list('media_id', flat=True))

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
# Interactive Analytics Dashboard View
# ======================================================================
@login_required
def analytics_dashboard(request):
    from .analytics_engine import generate_seaborn_heatmap, generate_plotly_scatter, generate_networkx_graph
    
    user = request.user
    watchlist_items = MovieWatchlist.objects.filter(user=user, media_type='movie')
    watchlist_movies = list(watchlist_items.values_list('media_id', flat=True))
    
    heatmap_base64 = generate_seaborn_heatmap()
    plotly_div_html = generate_plotly_scatter()
    network_base64 = generate_networkx_graph(watchlist_movies)
    
    context = {
        'heatmap_img': heatmap_base64,
        'plotly_div': plotly_div_html,
        'network_img': network_base64,
    }
    
    return render(request, 'core/analytics.html', context)


# ======================================================================
# Movie Detail Hub View
# ======================================================================
@login_required
def movie_detail_view(request, movie_id):
    api_key = settings.TMDB_API_KEY
    endpoint = (
        f"https://api.themoviedb.org/3/movie/{movie_id}"
        f"?api_key={api_key}"
        f"&language=en-US"
        f"&append_to_response=credits,videos,watch/providers,similar"
    )

    movie = {}
    cast = []
    trailer_key = None
    watch_providers = []
    similar_movies = []
    belongs_to_collection = None
    collection_movies = []
    collection_name = ""

    try:
        resp = requests.get(endpoint, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()

        poster_path = data.get('poster_path') or ''
        backdrop_path = data.get('backdrop_path') or ''
        
        usd_budget = data.get('budget', 0)
        usd_revenue = data.get('revenue', 0)
        inr_conversion_rate = 95.21
        
        movie = {
            'id':            data.get('id', movie_id),
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

        # 1. Define 'crew' first
        crew = credits_payload.get('crew', []) 
        
        # 2. Now you can find the director safely
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

        similar_payload = data.get('similar', {})
        raw_similar = similar_payload.get('results', [])
        for s in raw_similar[:5]:
            s_poster = s.get('poster_path') or ''
            encoded = urllib.parse.quote_plus(s.get('title', ''))
            similar_movies.append({
                'id':         s.get('id'),
                'movie_id':   s.get('id'),
                'title':      s.get('title', 'Unknown'),
                'vote_average': round(s.get('vote_average', 0.0), 1),
                'poster_url': (
                    f"https://image.tmdb.org/t/p/w300{s_poster}"
                    if s_poster else
                    'https://images.unsplash.com/photo-1542204172-e7052809f852?q=80&w=400&auto=format&fit=crop'
                ),
                'trailer_url': f"https://www.youtube.com/results?search_query={encoded}+official+trailer",
            })

        # Franchise / Collection Fetching
        belongs_to_collection = data.get('belongs_to_collection')
        if belongs_to_collection:
            collection_id = belongs_to_collection.get('id')
            collection_name = belongs_to_collection.get('name', '')
            collection_endpoint = f"https://api.themoviedb.org/3/collection/{collection_id}?api_key={api_key}&language=en-US"
            try:
                col_resp = requests.get(collection_endpoint, timeout=3.0)
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

    except requests.exceptions.RequestException as e:
        print(f"[MOVIE DETAIL] TMDB API request failed for movie_id={movie_id}: {e}")
        movie = {
            'id': movie_id, 'title': 'Data Unavailable', 'overview': '',
            'release_date': '', 'runtime': 0, 'vote_average': 0.0,
            'genres': [], 'poster_url': '', 'backdrop_url': '',
        }
    except Exception as e:
        print(f"[MOVIE DETAIL] Unexpected parsing error for movie_id={movie_id}: {e}")

    is_in_watchlist = MovieWatchlist.objects.filter(
        user=request.user, media_id=movie_id, media_type='movie'
    ).exists()

    from datetime import datetime
    release_date_str = movie.get('release_date', '')
    is_now_showing = False

    if release_date_str:
        try:
            release_date = datetime.strptime(release_date_str, '%Y-%m-%d').date()
            current_date = datetime.now().date()
            if release_date <= current_date and (current_date - release_date).days <= 45:
                is_now_showing = True
            elif release_date > current_date:
                is_now_showing = True
        except ValueError:
            pass

    # ── FETCH REVIEWS DATALAYER AND AUTHOR ACCESSORS ──
    reviews = Review.objects.filter(movie_id=movie_id).select_related('user')
    user_review = reviews.filter(user=request.user).first() if request.user.is_authenticated else None

    # Fetch watchlist IDs for bookmark toggle states
    watchlist_ids = []
    if request.user.is_authenticated:
        watchlist_ids = list(MovieWatchlist.objects.filter(
            user=request.user, media_type='movie'
        ).values_list('media_id', flat=True))

    # Fetch streaming links from Watchmode API via Cache (24 hour retention)
    from django.core.cache import cache
    from core.utils import get_streaming_links

    movie_title = movie.get('title', '')
    cache_key = f"watchmode_links_{movie_id}"
    streaming_links = cache.get(cache_key)

    if streaming_links is None:
        streaming_links = get_streaming_links(movie_title)
        cache.set(cache_key, streaming_links, 86400)

    # Map streaming links directly to the watch_providers items
    if watch_providers and streaming_links:
        for provider in watch_providers:
            provider_name = provider.get('name', '')
            norm_name = provider_name
            if "Prime" in provider_name or "Amazon" in provider_name:
                norm_name = "Prime Video"
            elif "Disney" in provider_name:
                norm_name = "Disney+"
            elif "Apple" in provider_name:
                norm_name = "Apple TV+"

            # Find matching link
            for service_name, url in streaming_links.items():
                if (service_name.lower() in provider_name.lower() or 
                    provider_name.lower() in service_name.lower() or
                    norm_name.lower() in service_name.lower()):
                    provider['web_url'] = url
                    break

    context = {
        'movie':           movie,
        'cast':            cast,
        'trailer_key':     trailer_key,
        'watch_providers': watch_providers,
        'streaming_links': streaming_links, # passed directly to context
        'similar_movies':  similar_movies,
        'recommendations': similar_movies, # mapped to recommendations
        'is_in_watchlist': is_in_watchlist,
        'is_now_showing':  is_now_showing,
        # 💎 INJECTED REVIEWS DATA CONTEXTS
        'reviews':         reviews,
        'user_review':     user_review,
        'director':        director,
        # Franchise Groups
        'belongs_to_collection': belongs_to_collection,
        'collection_movies':     collection_movies,
        'collection_name':       collection_name,
        'watchlist_ids':         watchlist_ids,
    }

    return render(request, 'core/movie_detail.html', context)


# ======================================================================
# TV Show Detail Hub View
# ======================================================================
@login_required
def tv_detail_view(request, series_id):
    api_key = settings.TMDB_API_KEY

    endpoint = (
        f"https://api.themoviedb.org/3/tv/{series_id}"
        f"?api_key={api_key}"
        f"&language=en-US"
        f"&append_to_response=credits,videos,watch/providers,similar"
    )

    tv_show = {}
    cast = []
    trailer_key = None
    watch_providers = []
    similar_shows = []

    try:
        resp = requests.get(endpoint, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()

        poster_path = data.get('poster_path') or ''
        backdrop_path = data.get('backdrop_path') or ''

        tv_show = {
            'id':                 data.get('id', series_id),
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

        similar_payload = data.get('similar', {})
        raw_similar = similar_payload.get('results', [])
        for s in raw_similar[:5]:
            s_poster = s.get('poster_path') or ''
            similar_shows.append({
                'id':           s.get('id'),
                'title':        s.get('name', 'Unknown'),
                'vote_average': round(s.get('vote_average', 0.0), 1),
                'poster_url': (
                    f"https://image.tmdb.org/t/p/w300{s_poster}"
                    if s_poster else
                    'https://images.unsplash.com/photo-1593305841991-05c297ba4575?q=80&w=400&auto=format&fit=crop'
                ),
            })

    except requests.exceptions.RequestException as e:
        print(f"[TV DETAIL] TMDB API request failed for series_id={series_id}: {e}")
        tv_show = {
            'id': series_id, 'title': 'Data Unavailable', 'overview': '',
            'first_air_date': '', 'number_of_seasons': 0, 'number_of_episodes': 0,
            'vote_average': 0.0, 'genres': [], 'poster_url': '', 'backdrop_url': '',
        }
    except Exception as e:
        print(f"[TV DETAIL] Unexpected parsing error for series_id={series_id}: {e}")

    is_in_watchlist = MovieWatchlist.objects.filter(
        user=request.user, media_id=series_id, media_type='tv'
    ).exists()

    # ── FETCH REVIEWS DATALAYER AND AUTHOR ACCESSORS ──
    reviews = MediaReview.objects.filter(media_id=series_id, media_type='tv').select_related('user')
    user_review = reviews.filter(user=request.user).first() if request.user.is_authenticated else None

    context = {
        'tv_show':         tv_show,
        'cast':            cast,
        'trailer_key':     trailer_key,
        'watch_providers': watch_providers,
        'similar_shows':   similar_shows,
        'is_in_watchlist': is_in_watchlist,
        # 💎 INJECTED REVIEWS DATA CONTEXTS
        'reviews':         reviews,
        'user_review':     user_review,
    }

    return render(request, 'core/tv_detail.html', context)

# ======================================================================
# Watchlist Hub Dashboard View
# ======================================================================
@login_required
def watchlist_hub_view(request):
    client = TMDBClient()
    db_items = MovieWatchlist.objects.filter(user=request.user).order_by("-id")

    watchlist_movies = []
    watchlist_tv = []

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

    # ── QUERY USER CUSTOM REVIEWS ──
    user_reviews = Review.objects.filter(user=request.user).order_by('-created_at')

    context = {
        'watchlist_movies': watchlist_movies,
        'watchlist_tv':     watchlist_tv,
        'user_reviews':     user_reviews,
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
    
    try:
        response = requests.get(url, params=params, timeout=5.0)
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
            print(f"[PROVIDER FEED] API returned status code {response.status_code} for provider {provider_id}")
            
    except Exception as e:
        print(f"[PROVIDER FEED] Error fetching for provider {provider_id}: {e}")
        
    return []


@login_required
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
        response = requests.get(person_url, headers=client.headers, timeout=5.0)
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
        response = requests.get(credits_url, headers=client.headers, timeout=5.0)
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
    import requests
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
        response = requests.get(url, headers=client.headers, params=params, timeout=5.0)
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
    Renders the dedicated search results page using TMDB search/multi query.
    """
    from .tmdb_api import TMDBClient
    import requests
    import urllib.parse
    
    query = request.GET.get('q', '').strip()
    movies = []
    tv_shows = []
    people = []
    
    if query:
        client = TMDBClient()
        url = f"{client.base_url}/search/multi"
        params = {
            'query': query,
            'language': 'en-US',
            'page': 1,
            'include_adult': 'false'
        }
        try:
            response = requests.get(url, headers=client.headers, params=params, timeout=5.0)
            if response.status_code == 200:
                results = response.json().get('results', [])
                for item in results:
                    media_type = item.get('media_type')
                    if media_type == 'movie':
                        poster_path = item.get('poster_path')
                        movies.append({
                            'id': item.get('id'),
                            'title': item.get('title') or item.get('original_title') or 'Unknown Movie',
                            'release_date': item.get('release_date', 'N/A'),
                            'poster_url': f"https://image.tmdb.org/t/p/w300{poster_path}" if poster_path else client.movie_fallback,
                            'vote_average': round(item.get('vote_average', 0.0), 1),
                            'overview': item.get('overview', '')
                        })
                    elif media_type == 'tv':
                        poster_path = item.get('poster_path')
                        tv_shows.append({
                            'id': item.get('id'),
                            'title': item.get('name') or item.get('original_name') or 'Unknown TV Show',
                            'first_air_date': item.get('first_air_date', 'N/A'),
                            'poster_url': f"https://image.tmdb.org/t/p/w300{poster_path}" if poster_path else client.tv_fallback,
                            'vote_average': round(item.get('vote_average', 0.0), 1),
                            'overview': item.get('overview', '')
                        })
                    elif media_type == 'person':
                        profile_path = item.get('profile_path')
                        known_for = [work.get('title') or work.get('name') for work in item.get('known_for', []) if work.get('title') or work.get('name')]
                        people.append({
                            'id': item.get('id'),
                            'name': item.get('name') or 'Unknown Person',
                            'profile_url': f"https://image.tmdb.org/t/p/w300{profile_path}" if profile_path else f"https://ui-avatars.com/api/?name={urllib.parse.quote_plus(item.get('name', 'Actor'))}",
                            'known_for': ", ".join(known_for[:3])
                        })
        except Exception as e:
            print(f"[SEARCH VIEW] Error querying TMDB API: {e}")

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
        'movies':           movies,
        'tv_shows':         tv_shows,
        'people':           people,
        'watchlist_movies': watchlist_movies,
        'watchlist_tv':     watchlist_tv,
    }
    return render(request, 'core/search_results.html', context)
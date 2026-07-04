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

from .models import UserProfile, MovieWatchlist, MediaReview
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
        poster_url = client.get_media_assets(media_id, media_type, timeout=2.5)
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
            movie_dict_path = os.path.join(settings.BASE_DIR, '..', 'movie_dict.pkl')
            movie_sim_path = os.path.join(settings.BASE_DIR, '..', 'similarity.pkl')
            tv_dict_path = os.path.join(settings.BASE_DIR, '..', 'tv_dict.pkl')
            tv_sim_path = os.path.join(settings.BASE_DIR, '..', 'tv_similarity.pkl')

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
    
    # ── FALLBACK LAYER 1: Watchlist is completely empty ──
    if not user_watchlist_ids:
        defaults = df.head(8).to_dict(orient='records')
        for d in defaults:
            title_text = d.get('title') or d.get('name') or 'Unknown Title'
            d['title'] = title_text
            d['poster_url'] = get_cached_poster(client, d[id_col], media_type)
            d['watch_link'] = client.get_streaming_or_theatre_links(title_text, media_type, False)
        return defaults

    # Find row indices of user's saved titles inside the catalog DataFrame
    watchlist_indices = df[df[id_col].isin(user_watchlist_ids)].index.tolist()
    
    # ── FALLBACK LAYER 2: Watchlist IDs do not match dataset records ──
    if not watchlist_indices:
        defaults = df.head(8).to_dict(orient='records')
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
# User Authentication & Registration View
# ======================================================================
@csrf_protect
def register_user(request):
    if request.user.is_authenticated:
        return redirect('for_you_feed')

    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            UserProfile.objects.create(user=user)
            login(request, user)
            return redirect('for_you_feed')
    else:
        form = UserCreationForm()
        
    return render(request, 'core/register.html', {'form': form})

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
        response = requests.get(url, headers=client.headers, timeout=1.0)
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
    
    context = {
        'watchlist_count': watchlist_items.count(),
        'saved_movies': saved_movies,
        'saved_tv_shows': saved_tv_shows,
        'saved_ids': saved_ids,
        'now_showing': now_showing,
        'recommended_movies': recommended_movies,
        'recommended_tv_shows': recommended_tv_shows,
        'spotlight_movie': spotlight_movie,
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
        })
        
    return render(request, 'core/for_you.html', context)

# ======================================================================
# Explore Movies View
# ======================================================================
@login_required
def explore_movies(request):
    import json
    client = TMDBClient()

    movies_df = pd.DataFrame()
    try:
        csv_path = os.path.join(settings.BASE_DIR, '..', 'movies.csv')
        movies_df = pd.read_csv(csv_path, usecols=['movie_id', 'title', 'cast', 'crew'])
    except Exception as e:
        print("Error reading movies.csv in explore view:", e)

    query = request.GET.get('q', '').strip()

    if not movies_df.empty:
        if query:
            q_lower = query.lower()

            def _names_from_json(cell):
                try:
                    entries = json.loads(cell) if isinstance(cell, str) else []
                    return ' '.join(e.get('name', '') for e in entries).lower()
                except (json.JSONDecodeError, TypeError, AttributeError):
                    return ''

            def _directors_from_crew(cell):
                try:
                    entries = json.loads(cell) if isinstance(cell, str) else []
                    return ' '.join(
                        e.get('name', '') for e in entries
                        if e.get('job', '').lower() == 'director'
                    ).lower()
                except (json.JSONDecodeError, TypeError, AttributeError):
                    return ''

            title_match    = movies_df['title'].str.lower().str.contains(q_lower, na=False)
            cast_series    = movies_df['cast'].apply(_names_from_json)
            crew_series    = movies_df['crew'].apply(_directors_from_crew)
            cast_match     = cast_series.str.contains(q_lower, na=False)
            director_match = crew_series.str.contains(q_lower, na=False)

            combined_mask  = title_match | cast_match | director_match
            filtered_df    = movies_df[combined_mask].drop_duplicates(subset=['movie_id'])
        else:
            filtered_df = movies_df

        movies_records = filtered_df[['movie_id', 'title']].to_dict(orient='records')
    else:
        movies_records = []

    paginator = Paginator(movies_records, 12)
    page_number = request.GET.get('page', 1)

    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)

    movies_on_page = []
    for movie in page_obj.object_list:
        movie['poster_url'] = get_cached_poster(client, movie['movie_id'], 'movie')
        movies_on_page.append(movie)

    watchlist_ids = list(MovieWatchlist.objects.filter(
        user=request.user, media_type='movie'
    ).values_list('media_id', flat=True))

    return render(request, 'core/explore_movies.html', {
        'movies':       movies_on_page,
        'page_obj':     page_obj,
        'watchlist_ids': watchlist_ids,
        'query':        query,
    })

# ======================================================================
# Explore TV Shows View
# ======================================================================
@login_required
def explore_tv(request):
    client = TMDBClient()
    
    tv_records_list = []
    try:
        csv_path = os.path.join(settings.BASE_DIR, '..', 'tv_shows.csv')
        tv_df = pd.read_csv(csv_path)
        tv_df = tv_df.drop_duplicates(subset=['name'])
        tv_raw = tv_df[['id', 'name', 'overview']].to_dict(orient='records')
        
        for rec in tv_raw:
            rec['title'] = rec.pop('name')
            tv_records_list.append(rec)
    except Exception as e:
        print("Error reading tv_shows.csv in explore view:", e)
        
    query = request.GET.get('q')
    if query:
        tv_records_list = [t for t in tv_records_list if query.lower() in t['title'].lower()]
        
    paginator = Paginator(tv_records_list, 12)
    page_number = request.GET.get('page', 1)
    
    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)
        
    tv_on_page = []
    for tv in page_obj.object_list:
        tv['poster_url'] = get_cached_poster(client, tv['id'], 'tv')
        tv_on_page.append(tv)
        
    watchlist_ids = list(MovieWatchlist.objects.filter(
        user=request.user, media_type='tv'
    ).values_list('media_id', flat=True))
    
    return render(request, 'core/explore_tv.html', {
        'tv_shows': tv_on_page,
        'page_obj': page_obj,
        'watchlist_ids': watchlist_ids
    })

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
    api_key = "41fc74ce5602882786e1e9d4933fdcc6"

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

    try:
        resp = requests.get(endpoint, timeout=5)
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
        for member in raw_cast[:6]:
            profile_path = member.get('profile_path') or ''
            cast.append({
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
                'title':      s.get('title', 'Unknown'),
                'vote_average': round(s.get('vote_average', 0.0), 1),
                'poster_url': (
                    f"https://image.tmdb.org/t/p/w300{s_poster}"
                    if s_poster else
                    'https://images.unsplash.com/photo-1542204172-e7052809f852?q=80&w=400&auto=format&fit=crop'
                ),
                'trailer_url': f"https://www.youtube.com/results?search_query={encoded}+official+trailer",
            })

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
    reviews = MediaReview.objects.filter(media_id=movie_id, media_type='movie').select_related('user')
    user_review = reviews.filter(user=request.user).first() if request.user.is_authenticated else None

    context = {
        'movie':           movie,
        'cast':            cast,
        'trailer_key':     trailer_key,
        'watch_providers': watch_providers,
        'similar_movies':  similar_movies,
        'is_in_watchlist': is_in_watchlist,
        'is_now_showing':  is_now_showing,
        # 💎 INJECTED REVIEWS DATA CONTEXTS
        'reviews':         reviews,
        'user_review':     user_review,
    }

    return render(request, 'core/movie_detail.html', context)


# ======================================================================
# TV Show Detail Hub View
# ======================================================================
@login_required
def tv_detail_view(request, series_id):
    api_key = "41fc74ce5602882786e1e9d4933fdcc6"

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
        resp = requests.get(endpoint, timeout=3)
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

    context = {
        'watchlist_movies': watchlist_movies,
        'watchlist_tv':     watchlist_tv,
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
    review = get_object_or_404(MediaReview, id=review_id)
    # ... auth check ...
    updated_text = request.POST.get('review_text', '').strip()
    if updated_text:
        # 💎 REMOVED escape()
        review.review_text = updated_text
        review.save()
        return JsonResponse({'success': True, 'message': 'Review updated successfully.'})
    return JsonResponse({'success': False, 'error': 'Review text cannot be blank.'}, status=400)

@login_required
@require_POST
def delete_media_review(request, review_id):
    """
    Syllabus Reference: Unit 9.2 (CRUD - Delete with Authorization Guard)
    """
    review = get_object_or_404(MediaReview, id=review_id)
    
    if review.user != request.user:
        return JsonResponse({'success': False, 'error': 'Unauthorized transaction request.'}, status=403)
        
    review.delete()
    return JsonResponse({'success': True, 'message': 'Review successfully scrubbed from catalog.'})
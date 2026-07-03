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
from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_protect
from django.http import JsonResponse
from django.conf import settings
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage

from .models import UserProfile, MovieWatchlist
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
    If no, executes client.get_media_assets() with a strict 500ms (0.5s) timeout
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
        # Enforce strict 500ms (0.5s) timeout block to avoid blocking Django server threads
        poster_url = client.get_media_assets(media_id, media_type, timeout=0.5)
        if poster_url:
            POSTER_CACHE[cache_key] = poster_url
            return poster_url
    except Exception as e:
        print(f"[CACHE ENGINE] Exception or timeout (>500ms) for key {cache_key}: {e}")
        
    return fallback_urls.get(media_type)


def load_ml_models():
    """
    Syllabus Reference: Units 4 & 5 Model Loading
    Caches the pre-computed Bag of Words similarity arrays
    and metadata dictionaries into server memory to optimize request speeds.
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
    - Aggregates the cosine similarity vectors for all titles currently in the user's watchlist.
    - S_agg = sum_{i in W} S_i, where W is the set of user-saved show indices and S_i is the similarity row.
    - Sorts all shows in descending order of similarity, filters out already saved elements,
      and returns the top 8 recommended items complete with pre-fetched TMDB poster images and watch links.
    """
    load_ml_models()
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
    
    # If user's watchlist is empty, return top 8 default items from the library catalog
    if not user_watchlist_ids:
        defaults = df.head(8).to_dict(orient='records')
        for d in defaults:
            # Use cached poster check instead of direct sync API calls
            d['poster_url'] = get_cached_poster(client, d[id_col], media_type)
            d['watch_link'] = client.get_streaming_or_theatre_links(d['title'], media_type, False)
        return defaults

    # Find row indices of user's saved titles inside the catalog DataFrame
    watchlist_indices = df[df[id_col].isin(user_watchlist_ids)].index.tolist()
    
    if not watchlist_indices:
        defaults = df.head(8).to_dict(orient='records')
        for d in defaults:
            d['poster_url'] = get_cached_poster(client, d[id_col], media_type)
            d['watch_link'] = client.get_streaming_or_theatre_links(d['title'], media_type, False)
        return defaults
        
    try:
        # Sum similarity columns for all user saved indices: S_agg = sum(sim_matrix[row_index])
        aggregated_sim = np.sum(sim_matrix[watchlist_indices], axis=0)
        
        # Sort scores in descending order
        sorted_indices = np.argsort(aggregated_sim)[::-1]
        
        # Filter out items that are already in the watchlist
        recommended_indices = [idx for idx in sorted_indices if idx not in watchlist_indices]
        
        # Extract top 8 recommendations
        top_indices = recommended_indices[:8]
        
        recommendations = df.iloc[top_indices].to_dict(orient='records')
        
        # Pre-fetch live posters utilizing memory caching
        for rec in recommendations:
            media_id = rec[id_col]
            rec['poster_url'] = get_cached_poster(client, media_id, media_type)
            rec['watch_link'] = client.get_streaming_or_theatre_links(rec['title'], media_type, False)
            
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
    """
    Syllabus Reference: Unit 9.2 (CRUD - Read), Units 4 & 5 (Model Inference),
    and Unit 7 (Real-Time Ingestion pipelines).
    Fetches real-time movie listings from the live TMDB Now Playing endpoint
    localized to India, applies safety timeouts, and registers URLs in POSTER_CACHE.
    """
    user = request.user
    client = TMDBClient()
    
    # Query database watchlist items
    watchlist_items = MovieWatchlist.objects.filter(user=user)
    
    # Extract saved movie IDs and TV IDs
    saved_movies = list(watchlist_items.filter(media_type='movie').values_list('media_id', flat=True))
    saved_tv_shows = list(watchlist_items.filter(media_type='tv').values_list('media_id', flat=True))
    saved_ids = list(watchlist_items.values_list('media_id', flat=True))
    
    # Dynamic spotlight movie definition (Unit 7 & 9)
    spotlight_movie = {
        'movie_id': 157336,
        'title': 'Interstellar',
        'overview': 'The adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances involved in an interstellar voyage.',
        'backdrop_url': 'https://image.tmdb.org/t/p/original/rAiYw1jKe6vS8v36ZasYwB66G6B.jpg'
    }
    
    # Live TMDB Ingestion Pipeline (Unit 7)
    now_showing = []
    url = f"{client.base_url}/movie/now_playing?language=en-US&region=IN&page=1"
    
    try:
        # Strict 1-second timeout constraint to prevent API hangs from blocking threads
        response = requests.get(url, headers=client.headers, timeout=1.0)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            for movie_data in results:
                movie_id = movie_data.get('id')
                title = movie_data.get('title')
                if not movie_id or not title:
                    continue
                
                # Resolve genre category strings
                genre_ids = movie_data.get('genre_ids', [])
                genre_names = [TMDB_GENRE_MAP.get(gid) for gid in genre_ids if TMDB_GENRE_MAP.get(gid)]
                genres_str = " | ".join(genre_names[:2]) or "Drama"
                
                # Fetch and store image path into global RAM memory cache
                poster_url = get_cached_poster(client, movie_id, 'movie')
                
                # Dynamic URL encoding for localized BookMyShow Ahmedabad redirect (Unit 7)
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
                
                # Prune and slice stream strictly to top 15 elements
                if len(now_showing) >= 15:
                    break
    except Exception as e:
        print(f"[TMDB LIVE INGESTION] Error fetching live now playing: {e}")
        
    # Robust Try/Except Fallback layer: Load pre-saved local link map if API is unreachable
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
    
    # Run vector inference models
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
# Syllabus Reference: Unit 9.2 (Pagination & Queryset Optimization)
# ======================================================================
@login_required
def explore_movies(request):
    import os
    from django.conf import settings
    import pandas as pd
    client = TMDBClient()
    
    movies_records = []
    try:
        csv_path = os.path.join(settings.BASE_DIR, '..', 'movies.csv')
        movies_df = pd.read_csv(csv_path)
        movies_records = movies_df[['movie_id', 'title']].to_dict(orient='records')
    except Exception as e:
        print("Error reading movies.csv in explore view:", e)
        
    query = request.GET.get('q')
    if query:
        movies_records = [m for m in movies_records if query.lower() in m['title'].lower()]
        
    # Wrap using Paginator: set limit of 12 items per page
    paginator = Paginator(movies_records, 12)
    page_number = request.GET.get('page', 1)
    
    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)
        
    # Utilize global RAM cache on the sliced items on this active page
    movies_on_page = []
    for movie in page_obj.object_list:
        movie['poster_url'] = get_cached_poster(client, movie['movie_id'], 'movie')
        movies_on_page.append(movie)
        
    watchlist_ids = list(MovieWatchlist.objects.filter(
        user=request.user, media_type='movie'
    ).values_list('media_id', flat=True))
    
    return render(request, 'core/explore_movies.html', {
        'movies': movies_on_page,
        'page_obj': page_obj,
        'watchlist_ids': watchlist_ids
    })

# ======================================================================
# Explore TV Shows View
# Syllabus Reference: Unit 9.2 (Pagination & Queryset Optimization)
# ======================================================================
@login_required
def explore_tv(request):
    import os
    from django.conf import settings
    import pandas as pd
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
        
    # Wrap using Paginator: set limit of 12 items per page
    paginator = Paginator(tv_records_list, 12)
    page_number = request.GET.get('page', 1)
    
    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)
        
    # Utilize global RAM cache on the active page TV shows
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
    """
    Dashboard view executing data analysis and sending visualization plots into HTML blocks.
    """
    from .analytics_engine import generate_seaborn_heatmap, generate_plotly_scatter, generate_networkx_graph
    
    user = request.user
    watchlist_items = MovieWatchlist.objects.filter(user=user, media_type='movie')
    watchlist_movies = list(watchlist_items.values_list('media_id', flat=True))
    
    # Generate dynamic visual plots
    heatmap_base64 = generate_seaborn_heatmap()
    plotly_div_html = generate_plotly_scatter()
    network_base64 = generate_networkx_graph(watchlist_movies)
    
    context = {
        'heatmap_img': heatmap_base64,
        'plotly_div': plotly_div_html,
        'network_img': network_base64,
    }
    
    return render(request, 'core/analytics.html', context)

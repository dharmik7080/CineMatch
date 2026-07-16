import requests
import hashlib
from django.core.cache import cache
from .tmdb_api import TMDBClient
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

_RESILIENT_SESSION = None

def get_resilient_session():
    """
    Syllabus Reference: Unit 7 (REST API Integration) & DevOps Best Practices
    Provides a requests.Session configured with exponential backoff retries.
    Handles 'Transient Faults' through 'Exponential Backoff' to ensure 'Resilient API Integration.'
    """
    global _RESILIENT_SESSION
    if _RESILIENT_SESSION is None:
        session = requests.Session()
        # Retries on 429 (Rate Limit) and standard transient server errors
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _RESILIENT_SESSION = session
    return _RESILIENT_SESSION

# TMDB Genre ID to Name Mapping dictionary
TMDB_GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Sci-Fi",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
    # TV-Specific Genres
    10759: "Action & Adventure", 10762: "Kids", 10763: "News", 10764: "Reality",
    10765: "Sci-Fi & Fantasy", 10766: "Soap", 10767: "Talk", 10768: "War & Politics"
}

def fetch_tmdb_catalog(endpoint_type="movie", list_type="popular", query=None, page=1):
    """
    Centralized utility to fetch catalog listings or search items from TMDB.
    Utilizes Django caching to limit API hits, and returns both mapped results 
    and total_pages from response.
    """
    # Create md5 hash of cache key parameters for safety
    cache_key_raw = f"tmdb_catalog_{endpoint_type}_{list_type}_{query}_{page}"
    cache_key = hashlib.md5(cache_key_raw.encode('utf-8')).hexdigest()
    
    cached_response = cache.get(cache_key)
    if cached_response:
        return cached_response

    client = TMDBClient()
    
    if query:
        # Search query execution
        url = f"{client.base_url}/search/{endpoint_type}"
        params = {
            'query': query,
            'language': 'en-US',
            'page': page,
            'include_adult': 'false'
        }
    else:
        # Catalog list execution (e.g. popular)
        url = f"{client.base_url}/{endpoint_type}/{list_type}"
        params = {
            'language': 'en-US',
            'page': page
        }
        
    try:
        response = get_resilient_session().get(url, headers=client.headers, params=params, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        results = data.get('results', [])
        total_pages = data.get('total_pages', 1)
        
        # Map variables to match core templates context expectation
        mapped_results = []
        for item in results:
            poster_path = item.get('poster_path')
            backdrop_path = item.get('backdrop_path')
            
            mapped_item = {
                'id': item.get('id'),
                'movie_id': item.get('id'), # Required for explore_movies template compatibility
                'title': item.get('title') or item.get('name') or item.get('original_title') or item.get('original_name') or 'Unknown Title',
                'release_date': item.get('release_date') or item.get('first_air_date') or 'N/A',
                'vote_average': round(item.get('vote_average', 0.0), 1),
                'overview': item.get('overview', ''),
                'poster_url': f"https://image.tmdb.org/t/p/w300{poster_path}" if poster_path else (client.movie_fallback if endpoint_type == 'movie' else client.tv_fallback),
                'backdrop_url': f"https://image.tmdb.org/t/p/original{backdrop_path}" if backdrop_path else '',
            }
            mapped_results.append(mapped_item)
            
        payload = {
            'results': mapped_results,
            'total_pages': total_pages
        }
        # Cache for 15 minutes (900 seconds)
        cache.set(cache_key, payload, timeout=900)
        return payload
    except Exception as e:
        print(f"[TMDB UTILS] Fetch error for {endpoint_type} (query={query}): {e}")
        return {'results': [], 'total_pages': 1}

def get_streaming_links(movie_title):
    """
    Queries Watchmode API to find streaming links for a movie title.
    1. Search for Watchmode ID of the movie title.
    2. Query sources for that Watchmode ID.
    3. Return dict of {service_name: web_url}.
    """
    from django.conf import settings
    api_key = getattr(settings, 'WATCHMODE_API_KEY', '')
    if not api_key:
        print("[WATCHMODE] API key is not configured.")
        return {}

    # Step 1: Search Watchmode ID
    search_url = "https://api.watchmode.com/v1/search/"
    params = {
        'apiKey': api_key,
        'search_field': 'name',
        'search_value': movie_title
    }
    try:
        response = get_resilient_session().get(search_url, params=params, timeout=15.0)
        response.raise_for_status()
        search_data = response.json()
        results = search_data.get('title_results', [])
        if not results:
            print(f"[WATCHMODE] No title results found for '{movie_title}'")
            return {}
        
        # Pick the first matching item ID
        watchmode_id = results[0].get('id')
        if not watchmode_id:
            return {}

        # Step 2: Query Sources for the Watchmode ID
        sources_url = f"https://api.watchmode.com/v1/title/{watchmode_id}/sources/"
        sources_params = {
            'apiKey': api_key
        }
        sources_response = get_resilient_session().get(sources_url, params=sources_params, timeout=15.0)
        sources_response.raise_for_status()
        sources_data = sources_response.json()
        
        # Step 3: Extract and filter sources
        streaming_links = {}
        for source in sources_data:
            name = source.get('name')
            web_url = source.get('web_url')
            if name and web_url:
                # Store the direct URL under the service name
                if name not in streaming_links:
                    streaming_links[name] = web_url
        
        return streaming_links
    except Exception as e:
        print(f"[WATCHMODE] Error fetching links for '{movie_title}': {e}")
        return {}


from datetime import datetime

def get_daily_trending_movies():
    """
    Fetches daily trending movies from TMDB /trending/movie/day API.
    Utilizes Cache-Aside pattern: stores (movies, timestamp) tuple.
    Returns: (trending_movies_list, last_updated_datetime)
    """
    from django.conf import settings
    api_key = getattr(settings, 'TMDB_API_KEY', '')
    if not api_key:
        print("[TMDB TRENDING WARNING] API Key not configured.")
        return [], None

    cache_key = "daily_trending_movies_cache"
    cached_payload = cache.get(cache_key)
    
    if cached_payload:
        # Return cached results and timestamp
        return cached_payload
        
    # Cache-Aside: Fetch fresh data from TMDB on cache miss
    url = "https://api.themoviedb.org/3/trending/movie/day"
    params = {
        'api_key': api_key,
        'language': 'en-US'
    }
    
    try:
        response = get_resilient_session().get(url, params=params, timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            trending_movies = []
            for item in results:
                poster_path = item.get('poster_path')
                trending_movies.append({
                    'id': item.get('id'),
                    'movie_id': item.get('id'),
                    'title': item.get('title', 'Unknown'),
                    'poster_url': f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://images.unsplash.com/photo-1542204172-e7052809f852?q=80&w=400&auto=format&fit=crop",
                    'vote_average': round(item.get('vote_average', 0.0), 1),
                    'release_date': item.get('release_date', '')
                })
            
            # Record current timestamp
            last_updated = datetime.now()
            
            payload = (trending_movies, last_updated)
            # Store in cache (daily trends refresh, cache for 12 hours = 43200 seconds)
            cache.set(cache_key, payload, 43200)
            return payload
            
    except Exception as e:
        print(f"[TMDB TRENDING ERROR] Failed to fetch daily trending movies: {e}")
        
    return [], None


def get_upcoming_movies():
    """
    Fetches upcoming movies from TMDB /movie/upcoming API.
    Caches result for 24 hours (86400 seconds).
    """
    from django.conf import settings
    from django.core.cache import cache
    import requests

    api_key = getattr(settings, 'TMDB_API_KEY', '')
    if not api_key:
        return []

    cache_key = "upcoming_movies_cache"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    url = "https://api.themoviedb.org/3/movie/upcoming"
    params = {
        'api_key': api_key,
        'language': 'en-US',
        'page': 1,
        'region': 'IN'
    }
    try:
        response = get_resilient_session().get(url, params=params, timeout=10.0)
        if response.status_code == 200:
            results = response.json().get('results', [])
            upcoming_movies = []
            for item in results:
                poster_path = item.get('poster_path')
                upcoming_movies.append({
                    'id': item.get('id'),
                    'movie_id': item.get('id'),
                    'title': item.get('title', 'Unknown'),
                    'poster_url': f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://images.unsplash.com/photo-1542204172-e7052809f852?q=80&w=400&auto=format&fit=crop",
                    'release_date': item.get('release_date', ''),
                    'vote_average': round(item.get('vote_average', 0.0), 1),
                })
            # Cache for 24 hours
            cache.set(cache_key, upcoming_movies, 86400)
            return upcoming_movies
    except Exception as e:
        print(f"[TMDB UPCOMING ERROR] Failed to fetch upcoming movies: {e}")

    return []


# ── STREAMING SERVICE PROVIDER NAME STANDARDIZATION & NORMALIZATION ──
SERVICE_MAP = {
    # Hotstar / Disney+ Hotstar / JioHotstar
    'hotstar': 'disney+',
    'disney+ hotstar': 'disney+',
    'disney hotstar': 'disney+',
    'disney plus hotstar': 'disney+',
    'disneyplus hotstar': 'disney+',
    'disney': 'disney+',
    'disney plus': 'disney+',
    'disneyplus': 'disney+',
    'jiohotstar': 'disney+',
    'jio hotstar': 'disney+',
    
    # Prime Video / Amazon Prime Video
    'prime video': 'prime video',
    'amazon prime video': 'prime video',
    'amazon prime': 'prime video',
    'amazon': 'prime video',
    'prime': 'prime video',
    
    # Apple TV / Apple TV+
    'apple tv': 'apple tv+',
    'apple tv+': 'apple tv+',
    'apple tv plus': 'apple tv+',
    'appletv': 'apple tv+',
    
    # Netflix
    'netflix': 'netflix',
}

def normalize_name(name):
    """
    Syllabus Reference: Unit 3.2 Feature Normalization & String Cleansing
    Standardizes a streaming provider/service name for robust keys comparison matching.
    """
    if not name or not isinstance(name, str):
        return ""
    
    # Lowercase & strip white spaces
    cleaned = name.lower().strip()
    
    # Use SERVICE_MAP if matched, otherwise default to the original name (fallback mode)
    return SERVICE_MAP.get(cleaned, name)

def fetch_omdb_data(imdb_id):
    """
    Syllabus Topic: Service separation and REST API integration (Unit 7)
    Fetches ratings and awards data from OMDb API using the movie/show IMDb ID.
    """
    if not imdb_id or not isinstance(imdb_id, str) or not imdb_id.startswith('tt'):
        return None
        
    from django.core.cache import cache
    cache_key = f"omdb_data_{imdb_id}"
    cached_val = cache.get(cache_key)
    if cached_val is not None:
        return cached_val
        
    from django.conf import settings
    import requests
    
    api_key = getattr(settings, 'OMDB_API_KEY', '')
    if not api_key:
        print("[OMDB] API Key is not configured in settings.")
        return None
        
    url = "http://www.omdbapi.com/"
    params = {
        'i': imdb_id,
        'apikey': api_key,
        'plot': 'full'
    }
    
    try:
        response = get_resilient_session().get(url, params=params, timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            if data.get('Response') == 'False':
                print(f"[OMDB] API returned error: {data.get('Error')}")
                return None
                
            # Extract Rotten Tomatoes score from Ratings array
            rt_score = 'N/A'
            for rating in data.get('Ratings', []):
                if rating.get('Source') == 'Rotten Tomatoes':
                    rt_score = rating.get('Value')
                    break
                    
            result = {
                'imdb_rating': data.get('imdbRating', 'N/A'),
                'rotten_tomatoes': rt_score,
                'awards': data.get('Awards', 'N/A'),
                'age_rating': data.get('Rated', 'N/A'),
                'full_plot': data.get('Plot', '')
            }
            cache.set(cache_key, result, 3600)
            return result
    except Exception as e:
        print(f"[OMDB] Exception in fetch_omdb_data for {imdb_id}: {e}")
        
    return None


def fetch_media_by_genre(genre_name, media_type="movie", page=1):
    """
    Syllabus Topic: Service separation and REST API discovery integration (Unit 7)
    Fetches movies or TV shows categorized under the specified genre from the TMDB discover API.
    """
    from django.conf import settings
    import requests
    
    # Normalize genre name synonyms based on media type
    genre_name_lower = genre_name.strip().lower()
    if media_type == "tv":
        if "sci-fi" in genre_name_lower or "fantasy" in genre_name_lower or "science fiction" in genre_name_lower or "scifi" in genre_name_lower or "sci_fi" in genre_name_lower:
            genre_name_lower = "sci-fi & fantasy"
        elif "action" in genre_name_lower or "adventure" in genre_name_lower:
            genre_name_lower = "action & adventure"
        elif "war" in genre_name_lower or "politics" in genre_name_lower:
            genre_name_lower = "war & politics"
    else: # movie
        if "science fiction" in genre_name_lower or "scifi" in genre_name_lower or "sci-fi" in genre_name_lower or "sci_fi" in genre_name_lower:
            genre_name_lower = "sci-fi"
        elif "tv" in genre_name_lower or "television" in genre_name_lower:
            genre_name_lower = "tv movie"

    genre_id = None
    target_genre_name = genre_name
    
    for gid, gname in TMDB_GENRE_MAP.items():
        if gname.lower() == genre_name_lower:
            genre_id = gid
            target_genre_name = gname
            break

    if not genre_id:
        return [], target_genre_name, 1

    api_key = getattr(settings, 'TMDB_API_KEY', '')
    url = f"https://api.themoviedb.org/3/discover/{media_type}"
    params = {
        'api_key': api_key,
        'language': 'en-US',
        'sort_by': 'popularity.desc',
        'include_adult': 'false',
        'page': page,
        'with_genres': int(genre_id)
    }

    records = []
    total_pages = 1
    
    try:
        response = get_resilient_session().get(url, params=params, timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            raw_results = data.get('results', [])
            for item in raw_results:
                item_id = item.get('id')
                if item_id:
                    poster_path = item.get('poster_path')
                    poster_url = f"https://image.tmdb.org/t/p/w300{poster_path}" if poster_path else "https://images.unsplash.com/photo-1542204172-e7052809f852?q=80&w=400&auto=format&fit=crop"
                    
                    title = item.get('title') if media_type == 'movie' else item.get('name')
                    release_date = item.get('release_date') if media_type == 'movie' else item.get('first_air_date')
                    
                    records.append({
                        'id': int(item_id),
                        'media_id': int(item_id),
                        'movie_id': int(item_id),  # Template compatibility
                        'title': title or 'Unknown Title',
                        'name': title or 'Unknown Title',
                        'poster_url': poster_url,
                        'vote_average': round(item.get('vote_average', 0.0), 1),
                        'release_date': release_date or '',
                        'first_air_date': release_date or ''
                    })
            total_pages = min(data.get('total_pages', 1), 500)
    except Exception as e:
        print(f"[FETCH GENRE ERROR] Failed discover query for genre={genre_name}: {e}")
        
    return records, target_genre_name, total_pages

import requests
import hashlib
from django.core.cache import cache
from .tmdb_api import TMDBClient

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
        response = requests.get(url, headers=client.headers, params=params, timeout=5.0)
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

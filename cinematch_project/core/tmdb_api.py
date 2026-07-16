"""
CineMatch Project - Phase 3: Data Ingestion & API Connector
Sub-Phase 3.2: TMDB REST API Ingestion Module

Syllabus Reference:
- Unit 7: Web Scraping, APIs & Data Ingestion (focusing on REST APIs using requests, JSON handling, and secure Authorization Tokens)
"""

import requests
from urllib.parse import quote_plus
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

# Bearer Authorization Token from app.py to authorize REST API calls
TMDB_DEFAULT_BEARER_TOKEN = (
    "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI0MWZjNzRjZTU2MDI4ODI3ODZlMWU5ZDQ5MzNmZGNjNiIsIm5iZiI6MTcyMjkzOTI3OC4xNzY5NjUs"
    "InN1YiI6IjY2YjFmNTNjOGUxOGRjNTA1YTllNjhjNiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.UYyGr5Cw__O1k0SK-Blyreukq4qLyS7e4nwDhPjxk5A"
)

class TMDBClient:
    """
    Client class for interacting with the TMDB REST API.
    Manages HTTP request headers, token-based authorization, JSON payload parsing, and resource mapping.
    """
    def __init__(self, token=None):
        # Syllabus Topic: Secure Token-based API Authorization (Unit 7)
        self.api_token = token or TMDB_DEFAULT_BEARER_TOKEN
        self.base_url = "https://api.themoviedb.org/3"
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_token}"
        }
        self.image_base_url = "https://image.tmdb.org/t/p/w500"
        self.logo_base_url = "https://image.tmdb.org/t/p/original"
        
        # Fallback image links (Unsplash) to handle cases where no poster is returned
        self.movie_fallback = "https://images.unsplash.com/photo-1542204172-e7052809f852?q=80&w=400&auto=format&fit=crop"
        self.tv_fallback = "https://images.unsplash.com/photo-1593305841991-05c297ba4575?q=80&w=400&auto=format&fit=crop"

        # Initialize requests session with retry strategy and exponential backoff
        from core.utils import get_resilient_session
        self.session = get_resilient_session()

        # Dedicated fast, non-retrying session specifically for non-critical assets (posters)
        # to prevent thread starvation and page hanging when TMDB is offline.
        self.asset_session = requests.Session()
        self.asset_session.mount('http://', HTTPAdapter(max_retries=0))
        self.asset_session.mount('https://', HTTPAdapter(max_retries=0))

    def get_media_assets(self, media_id, media_type, timeout=3.0):
        """
        Syllabus Topic: REST API consumption and JSON data extraction (Unit 7)
        Fetches the primary poster path or backdrop path for a movie or TV show.
        - media_id: unique TMDB tracking ID
        - media_type: 'movie' or 'tv'
        """
        if media_type not in ['movie', 'tv']:
            return self.movie_fallback

        from django.core.cache import cache
        cache_key = f"tmdb_media_assets_{media_type}_{media_id}"
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        url = f"{self.base_url}/{media_type}/{media_id}?language=en-US"
        
        try:
            # Send HTTP GET requests with custom timeout parameter using non-retrying asset session
            response = self.asset_session.get(url, headers=self.headers, timeout=timeout)
            
            # Inspect HTTP status codes
            if response.status_code == 200:
                # Ingest JSON payload
                data = response.json()
                poster = data.get('poster_path')
                backdrop = data.get('backdrop_path')
                
                # Build complete secure URL to static resources
                result = None
                if poster:
                    result = f"{self.image_base_url}{poster}"
                elif backdrop:
                    result = f"{self.image_base_url}{backdrop}"
                
                if result:
                    cache.set(cache_key, result, 86400)
                    return result
            
            print(f"[TMDB] Failed to load assets for {media_type} ID {media_id}. Status: {response.status_code}")
        except requests.RequestException as e:
            print(f"[TMDB] Network error in get_media_assets for {media_type} ID {media_id}: {e}")
            
        # Return fallback cover on error
        return self.tv_fallback if media_type == 'tv' else self.movie_fallback

    def get_where_to_watch(self, media_id, media_type):
        """
        Syllabus Topic: Nested JSON data manipulation (Unit 7)
        Queries TMDB's watch provider endpoint and filters for providers located in India ('IN').
        Retrieves streaming networks (e.g. Netflix, Apple TV) and their logo icon resources.
        """
        if media_type not in ['movie', 'tv']:
            return []

        from django.core.cache import cache
        cache_key = f"tmdb_where_to_watch_{media_type}_{media_id}"
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        url = f"{self.base_url}/{media_type}/{media_id}/watch/providers"
        providers = []
        
        try:
            response = self.session.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', {})
                
                # Fetch provider listing for India (IN region code)
                in_region = results.get('IN', {})
                
                # Iterate across flatrate, rent, and buy options
                subscription_types = ['flatrate', 'rent', 'buy']
                seen_names = set()
                
                for subtype in subscription_types:
                    for provider_data in in_region.get(subtype, []):
                        name = provider_data.get('provider_name')
                        if name and name not in seen_names:
                            seen_names.add(name)
                            logo_path = provider_data.get('logo_path')
                            logo_url = f"{self.logo_base_url}{logo_path}" if logo_path else ""
                            providers.append({
                                'name': name,
                                'logo': logo_url,
                                'type': subtype.capitalize()
                            })
                cache.set(cache_key, providers, 86400)
                return providers
        except requests.RequestException as e:
            print(f"[TMDB] Network error in get_where_to_watch for {media_type} ID {media_id}: {e}")
            
        return providers

    def get_streaming_or_theatre_links(self, title, media_type, is_now_showing=False):
        """
        Syllabus Topic: URL Encoding and Dynamic Link Construction (Unit 7)
        Creates external ticket booking or streaming query links using title string encoding.
        """
        if not title:
            return "#"
            
        # Clean URL encoding of title characters
        encoded_title = quote_plus(title)
        
        if is_now_showing:
            # Dynamic location-aware BookMyShow movie query link
            return f"https://in.bookmyshow.com/explore/movies?search={encoded_title}"
        else:
            # Google Search query shortcut link for streaming
            return f"https://www.google.com/search?q=watch+{encoded_title}+online"

    def get_similar_movies(self, movie_id):
        """
        Fetches similar movies, filters by popularity > 5.0 and vote_average > 6.0,
        and falls back to popular movies of the same genre if < 3 results are found.
        """
        from django.core.cache import cache
        cache_key = f"tmdb_similar_movies_{movie_id}"
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        from django.conf import settings
        api_key = getattr(settings, 'TMDB_API_KEY', '')
        if not api_key:
            print("[TMDB] API key is not configured in settings.")
            return []

        # Step 1: Query Similar movies
        similar_url = f"{self.base_url}/movie/{movie_id}/similar"
        params = {
            'api_key': api_key,
            'language': 'en-US',
            'page': 1
        }
        
        filtered_movies = []
        seen_ids = {int(movie_id)}  # Prevent including the current movie itself

        try:
            response = self.session.get(similar_url, params=params, timeout=10.0)
            if response.status_code == 200:
                results = response.json().get('results', [])
                for item in results:
                    pop = item.get('popularity', 0.0)
                    vote = item.get('vote_average', 0.0)
                    m_id = item.get('id')
                    
                    # Popularity & Rating Filtering
                    if pop > 5.0 and vote > 6.0 and m_id not in seen_ids:
                        seen_ids.add(m_id)
                        poster_path = item.get('poster_path')
                        poster_url = f"{self.image_base_url}{poster_path}" if poster_path else self.movie_fallback
                        filtered_movies.append({
                            'id': m_id,
                            'title': item.get('title'),
                            'poster_url': poster_url,
                            'release_date': item.get('release_date', ''),
                            'vote_average': round(vote, 1),
                        })
        except Exception as e:
            print(f"[TMDB WARNING] Error fetching similar movies for movie_id={movie_id}: {e}")

        # Step 2: Hybrid Fallback if filtered list contains fewer than 3 movies
        if len(filtered_movies) < 3:
            print(f"[TMDB INFO] Similar results count ({len(filtered_movies)}) is less than 3. Triggering genre fallback...")
            try:
                # 2a. Get movie details to extract the primary genre ID
                details_url = f"{self.base_url}/movie/{movie_id}"
                det_resp = self.session.get(details_url, params={'api_key': api_key}, timeout=5.0)
                if det_resp.status_code == 200:
                    genres = det_resp.json().get('genres', [])
                    if genres:
                        primary_genre_id = genres[0].get('id')
                        
                        # 2b. Query discover/movie for popular movies in this genre
                        discover_url = f"{self.base_url}/discover/movie"
                        disc_params = {
                            'api_key': api_key,
                            'with_genres': primary_genre_id,
                            'sort_by': 'popularity.desc',
                            'language': 'en-US',
                            'page': 1
                        }
                        disc_resp = self.session.get(discover_url, params=disc_params, timeout=5.0)
                        if disc_resp.status_code == 200:
                            disc_results = disc_resp.json().get('results', [])
                            for item in disc_results:
                                m_id = item.get('id')
                                if m_id not in seen_ids:
                                    seen_ids.add(m_id)
                                    poster_path = item.get('poster_path')
                                    poster_url = f"{self.image_base_url}{poster_path}" if poster_path else self.movie_fallback
                                    filtered_movies.append({
                                        'id': m_id,
                                        'title': item.get('title'),
                                        'poster_url': poster_url,
                                        'release_date': item.get('release_date', ''),
                                        'vote_average': round(item.get('vote_average', 0.0), 1),
                                    })
                                    # Limit the list to 5 or 6 recommendations total
                                    if len(filtered_movies) >= 6:
                                        break
            except Exception as ex:
                print(f"[TMDB WARNING] Fallback genre discovery failed for movie_id={movie_id}: {ex}")

        cache.set(cache_key, filtered_movies, 86400)
        return filtered_movies

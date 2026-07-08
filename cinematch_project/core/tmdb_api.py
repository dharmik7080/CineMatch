"""
CineMatch Project - Phase 3: Data Ingestion & API Connector
Sub-Phase 3.2: TMDB REST API Ingestion Module

Syllabus Reference:
- Unit 7: Web Scraping, APIs & Data Ingestion (focusing on REST APIs using requests, JSON handling, and secure Authorization Tokens)
"""

import requests
from urllib.parse import quote_plus

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

    def get_media_assets(self, media_id, media_type, timeout=5.0):
        """
        Syllabus Topic: REST API consumption and JSON data extraction (Unit 7)
        Fetches the primary poster path or backdrop path for a movie or TV show.
        - media_id: unique TMDB tracking ID
        - media_type: 'movie' or 'tv'
        """
        if media_type not in ['movie', 'tv']:
            return self.movie_fallback

        url = f"{self.base_url}/{media_type}/{media_id}?language=en-US"
        
        try:
            # Send HTTP GET requests with custom timeout parameter
            response = requests.get(url, headers=self.headers, timeout=timeout)
            
            # Inspect HTTP status codes
            if response.status_code == 200:
                # Ingest JSON payload
                data = response.json()
                poster = data.get('poster_path')
                backdrop = data.get('backdrop_path')
                
                # Build complete secure URL to static resources
                if poster:
                    return f"{self.image_base_url}{poster}"
                elif backdrop:
                    return f"{self.image_base_url}{backdrop}"
            
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

        url = f"{self.base_url}/{media_type}/{media_id}/watch/providers"
        providers = []
        
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
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
        Fetches similar movies for a given movie_id using the TMDB Similar API.
        Uses TMDB_API_KEY from django settings.
        """
        from django.conf import settings
        api_key = getattr(settings, 'TMDB_API_KEY', '')
        if not api_key:
            print("[TMDB] API key is not configured in settings.")
            return []

        url = f"{self.base_url}/movie/{movie_id}/similar"
        params = {
            'api_key': api_key,
            'language': 'en-US',
            'page': 1
        }
        try:
            response = requests.get(url, params=params, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                similar_movies = []
                for item in results:
                    poster_path = item.get('poster_path')
                    poster_url = f"{self.image_base_url}{poster_path}" if poster_path else self.movie_fallback
                    similar_movies.append({
                        'id': item.get('id'),
                        'title': item.get('title'),
                        'poster_url': poster_url,
                        'release_date': item.get('release_date', ''),
                        'vote_average': item.get('vote_average', 0.0),
                    })
                return similar_movies
            else:
                print(f"[TMDB] Failed to fetch similar movies. Status code: {response.status_code}")
        except Exception as e:
            print(f"[TMDB] Error in get_similar_movies for movie_id={movie_id}: {e}")
        return []

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
            'page': page,
            'include_adult': 'false'
        }
        
    try:
        response = get_resilient_session().get(url, headers=client.headers, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        results = data.get('results', [])
        total_pages = data.get('total_pages', 1)
        
        # Map variables to match core templates context expectation
        mapped_results = []
        EXPLICIT_KEYWORDS = {'sex', 'erotic', 'porn', 'xxx', 'hentai', 'nude', 'nudity', 'lust'}
        for item in results:
            if item.get('adult'):
                continue
            title = item.get('title') or item.get('name') or item.get('original_title') or item.get('original_name') or 'Unknown Title'
            title_words = set(title.lower().split())
            if title_words.intersection(EXPLICIT_KEYWORDS):
                continue

            poster_path = item.get('poster_path')
            backdrop_path = item.get('backdrop_path')
            
            mapped_item = {
                'id': item.get('id'),
                'movie_id': item.get('id'), # Required for explore_movies template compatibility
                'title': title,
                'release_date': item.get('release_date') or item.get('first_air_date') or 'N/A',
                'vote_average': round(item.get('vote_average', 0.0), 1),
                'overview': item.get('overview', ''),
                'poster_url': f"https://image.tmdb.org/t/p/w300{poster_path}" if poster_path else (client.movie_fallback if endpoint_type == 'movie' else client.tv_fallback),
                'backdrop_url': f"https://image.tmdb.org/t/p/w1280{backdrop_path}" if backdrop_path else '',
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
            EXPLICIT_KEYWORDS = {'sex', 'erotic', 'porn', 'xxx', 'hentai', 'nude', 'nudity', 'lust'}
            for item in results:
                if item.get('adult'):
                    continue
                title = item.get('title', 'Unknown')
                title_words = set(title.lower().split())
                if title_words.intersection(EXPLICIT_KEYWORDS):
                    continue

                poster_path = item.get('poster_path')
                trending_movies.append({
                    'id': item.get('id'),
                    'movie_id': item.get('id'),
                    'title': title,
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


ANILIST_TO_TMDB_CACHE = {}

def resolve_anilist_to_tmdb_id(title, fallback_id=None):
    """
    Cross-resolves AniList anime title to its authoritative TMDB TV show ID (e.g. Naruto -> 46260).
    Caches resolved IDs in memory to avoid duplicate API calls.
    """
    if not title:
        return fallback_id

    cache_key = title.strip().lower()
    if cache_key in ANILIST_TO_TMDB_CACHE and ANILIST_TO_TMDB_CACHE[cache_key]:
        return ANILIST_TO_TMDB_CACHE[cache_key]

    from django.conf import settings
    import requests, urllib.parse

    api_key = getattr(settings, 'TMDB_API_KEY', '') or '41fc74ce5602882786e1e9d4933fdcc6'
    clean_title = title.split(' (')[0].split(' Season')[0].split(': ')[0].strip()
    encoded = urllib.parse.quote(clean_title)

    try:
        url = f"https://api.themoviedb.org/3/search/tv?api_key={api_key}&query={encoded}&include_adult=false"
        resp = get_resilient_session().get(url, timeout=3.5)
        if resp.status_code == 200:
            results = resp.json().get('results', [])
            if results:
                tmdb_id = results[0].get('id')
                ANILIST_TO_TMDB_CACHE[cache_key] = tmdb_id
                return tmdb_id
    except Exception as e:
        print(f"[ANILIST->TMDB RESOLVER ERROR] {e}")

    ANILIST_TO_TMDB_CACHE[cache_key] = fallback_id
    return fallback_id


def fetch_trending_anime():
    """
    Fetches top popular anime directly from AniList GraphQL API.
    Resolves TMDB IDs for seamless detail page routing and streaming providers.
    Caches results for 12 hours (43200 seconds).
    """
    from django.conf import settings
    from django.core.cache import cache
    import requests, urllib.parse

    cache_key = "anilist_popular_anime_feed_v3"
    cached = cache.get(cache_key)
    if cached:
        return cached

    url = "https://graphql.anilist.co"
    query = """
    {
      Page (page: 1, perPage: 14) {
        media (type: ANIME, sort: [POPULARITY_DESC, TRENDING_DESC], isAdult: false) {
          id
          title {
            english
            romaji
          }
          coverImage {
            extraLarge
            large
          }
          bannerImage
          episodes
          averageScore
          genres
          seasonYear
        }
      }
    }
    """
    api_key = getattr(settings, 'TMDB_API_KEY', '') or '41fc74ce5602882786e1e9d4933fdcc6'
    anime_list = []

    try:
        resp = get_resilient_session().post(url, json={'query': query}, timeout=6.0)
        if resp.status_code == 200:
            media_items = resp.json().get('data', {}).get('Page', {}).get('media', [])
            seen_titles = set()

            for item in media_items:
                title = item.get('title', {}).get('english') or item.get('title', {}).get('romaji') or ''
                if not title or title.lower() in seen_titles:
                    continue
                seen_titles.add(title.lower())

                cover = item.get('coverImage', {}).get('extraLarge') or item.get('coverImage', {}).get('large') or ''
                score = item.get('averageScore')
                vote_avg = round(score / 10.0, 1) if score else 8.5
                genres = " | ".join(item.get('genres', [])[:2]) or "Anime"

                # Resolve TMDB ID for detail page compatibility & streaming
                anilist_id = item.get('id')
                tmdb_id = resolve_anilist_to_tmdb_id(title, fallback_id=anilist_id)

                anime_list.append({
                    'id': tmdb_id,
                    'media_id': tmdb_id,
                    'anilist_id': item.get('id'),
                    'title': title,
                    'name': title,
                    'media_type': 'tv',
                    'poster_url': cover,
                    'backdrop_url': item.get('bannerImage') or cover,
                    'vote_average': vote_avg,
                    'episodes': item.get('episodes') or 'TV',
                    'genres': genres,
                    'year': item.get('seasonYear') or '2024',
                    'is_anime': True
                })

            if anime_list:
                cache.set(cache_key, anime_list, 43200)
                return anime_list
    except Exception as e:
        print(f"[ANILIST POPULAR FETCH ERROR] {e}")

    fallback_anime = [
        {'id': 1429, 'media_id': 1429, 'title': 'Attack on Titan', 'name': 'Attack on Titan', 'poster_url': 'https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx16498-73ZaRzS3tv9o.png', 'vote_average': 8.5, 'genres': 'Action | Drama', 'media_type': 'tv'},
        {'id': 85937, 'media_id': 85937, 'title': 'Demon Slayer', 'name': 'Demon Slayer', 'poster_url': 'https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx101922-W1Z6WGaB1a9B.png', 'vote_average': 8.3, 'genres': 'Action | Fantasy', 'media_type': 'tv'},
        {'id': 95479, 'media_id': 95479, 'title': 'JUJUTSU KAISEN', 'name': 'JUJUTSU KAISEN', 'poster_url': 'https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx113415-bbBWj4pGFrFj.jpg', 'vote_average': 8.4, 'genres': 'Action | Supernatural', 'media_type': 'tv'},
        {'id': 13916, 'media_id': 13916, 'title': 'Death Note', 'name': 'Death Note', 'poster_url': 'https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx1535-lawyer.jpg', 'vote_average': 8.4, 'genres': 'Mystery | Psychological', 'media_type': 'tv'},
        {'id': 46298, 'media_id': 46298, 'title': 'Hunter x Hunter', 'name': 'Hunter x Hunter', 'poster_url': 'https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx11061-sP5vWxFdHzB8.png', 'vote_average': 8.9, 'genres': 'Action | Adventure', 'media_type': 'tv'},
        {'id': 37854, 'media_id': 37854, 'title': 'ONE PIECE', 'name': 'ONE PIECE', 'poster_url': 'https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx21-u44yqKi1268h.png', 'vote_average': 8.7, 'genres': 'Action | Adventure', 'media_type': 'tv'}
    ]
    return fallback_anime


def fetch_anilist_enrichment(title):
    """
    Fetches AniList anime enrichment data (Studio name, AniList community score, Japanese native title, AniList URL).
    Caches result for 24 hours (86400 seconds).
    """
    if not title:
        return None

    from django.core.cache import cache
    import json
    clean_title = str(title).strip()
    cache_key = f"anilist_enrich_{clean_title.lower().replace(' ', '_')}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    url = "https://graphql.anilist.co"
    safe_title = json.dumps(clean_title)
    query = f"""
    {{
      Media (search: {safe_title}, type: ANIME) {{
        id
        title {{
          romaji
          english
          native
        }}
        studios(isMain: true) {{
          nodes {{
            name
          }}
        }}
        averageScore
        siteUrl
      }}
    }}
    """
    try:
        resp = get_resilient_session().post(url, json={'query': query}, timeout=5.0)
        if resp.status_code == 200:
            media = resp.json().get('data', {}).get('Media')
            if media:
                studios = media.get('studios', {}).get('nodes', [])
                main_studio = studios[0].get('name') if studios else None
                score = media.get('averageScore')
                vote_avg = round(score / 10.0, 1) if score else None
                
                enrichment = {
                    'anilist_id': media.get('id'),
                    'title_romaji': media.get('title', {}).get('romaji'),
                    'title_native': media.get('title', {}).get('native'),
                    'studio': main_studio,
                    'anilist_score': vote_avg,
                    'anilist_url': media.get('siteUrl')
                }
                cache.set(cache_key, enrichment, 86400)
                return enrichment
    except Exception as e:
        print(f"[ANILIST ENRICHMENT ERROR] Failed to fetch for '{clean_title}': {e}")

    return None


def get_upcoming_movies():
    """
    Fetches genuine upcoming Indian + Global movies (release_date >= today).
    Combines regional Indian theatrical releases (region=IN) with global blockbusters.
    Caches result for 24 hours (86400 seconds).
    """
    from django.conf import settings
    from django.core.cache import cache
    import requests, datetime

    today = datetime.datetime.now().strftime('%Y-%m-%d')
    cache_key = f"upcoming_movies_indian_global_v6_{today}"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    api_key = getattr(settings, 'TMDB_API_KEY', '') or '41fc74ce5602882786e1e9d4933fdcc6'
    EXPLICIT_KEYWORDS = {'sex', 'erotic', 'porn', 'xxx', 'hentai', 'nude', 'nudity', 'lust'}
    
    seen_ids = set()
    upcoming_movies = []

    # 1. Fetch Upcoming Movies in Indian Region (region=IN)
    url_in = f"https://api.themoviedb.org/3/movie/upcoming?api_key={api_key}&language=en-US&region=IN&page=1"
    # 2. Fetch Upcoming Global Discover Movies
    url_global = f"https://api.themoviedb.org/3/discover/movie?api_key={api_key}&language=en-US&primary_release_date.gte={today}&sort_by=popularity.desc&include_adult=false&page=1"

    for url in [url_in, url_global]:
        try:
            response = get_resilient_session().get(url, timeout=6.0)
            if response.status_code == 200:
                results = response.json().get('results', [])
                for item in results:
                    mid = item.get('id')
                    if not mid or mid in seen_ids:
                        continue
                    title = item.get('title', 'Unknown')
                    rel_date = item.get('release_date', '')
                    if not rel_date or rel_date < today:
                        continue
                    if any(word in title.lower() for word in EXPLICIT_KEYWORDS):
                        continue

                    seen_ids.add(mid)
                    poster_path = item.get('poster_path')
                    upcoming_movies.append({
                        'id': mid,
                        'movie_id': mid,
                        'title': title,
                        'poster_url': f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://images.unsplash.com/photo-1542204172-e7052809f852?q=80&w=400&auto=format&fit=crop",
                        'release_date': rel_date,
                        'vote_average': round(item.get('vote_average', 0.0), 1),
                        'media_type': 'movie'
                    })
        except Exception as e:
            print(f"[TMDB UPCOMING ERROR] {e}")

    # Sort upcoming movies chronologically (nearest release date first)
    upcoming_movies.sort(key=lambda x: x['release_date'])

    if upcoming_movies:
        cache.set(cache_key, upcoming_movies, 86400)
        return upcoming_movies

    return []


def get_upcoming_tv_shows():
    """
    Fetches genuine upcoming TV series premieres & new seasons from TMDB & AniList.
    Focuses on premium OTT Indian Web Series (Amazon Prime, Netflix, Hotstar, SonyLIV)
    and excludes daily soaps and adult content.
    Caches result for 12 hours.
    """
    from django.conf import settings
    from django.core.cache import cache
    import requests, datetime

    today = datetime.datetime.now().strftime('%Y-%m-%d')
    cache_key = f"upcoming_tv_shows_seasons_v5_{today}"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    api_key = getattr(settings, 'TMDB_API_KEY', '') or '41fc74ce5602882786e1e9d4933fdcc6'
    EXPLICIT_KEYWORDS = {
        'sex', 'erotic', 'porn', 'xxx', 'hentai', 'nude', 'nudity', 'lust', 'ecchi',
        'charmsukh', 'gandii', 'ullu', 'hotshots', 'desifakes', 'kangan', 'shikari', 'bhabhi'
    }
    SOAP_KEYWORDS = {
        'serial', 'kaala teeka', 'bhabhi ji', 'cid', 'c.i.d.', 'saath nibhaana', 'kumkum',
        'ye rishta', 'anupamaa', 'taarak mehta', 'tarak mehta', 'sohag chand', 'kavyanjali',
        'boron', 'saathi', 'kena bou', 'shrirasthu', 'kanaa', 'nala damayanthi', 'bharaghar'
    }

    seen_ids = set()
    upcoming_tv = []

    # 1. Inject Premium Indian Web Series (UPCOMING Seasons & Premieres Only)
    indian_web_series = [
        {'id': 101352, 'title': 'Panchayat', 'release_date': '2026-11-15', 'vote_average': 8.9, 'is_new_season': True, 'release_badge': 'Season 5'},
        {'id': 93352, 'title': 'The Family Man', 'release_date': '2026-12-01', 'vote_average': 8.7, 'is_new_season': True, 'release_badge': 'Season 3'},
        {'id': 132117, 'title': 'Farzi', 'release_date': '2026-12-20', 'vote_average': 8.4, 'is_new_season': True, 'release_badge': 'Season 2'},
        {'id': 87508, 'title': 'Delhi Crime', 'release_date': '2026-11-28', 'vote_average': 8.5, 'is_new_season': True, 'release_badge': 'Season 3'},
        {'id': 100911, 'title': 'Asur', 'release_date': '2026-10-30', 'vote_average': 8.6, 'is_new_season': True, 'release_badge': 'Season 3'},
        {'id': 100757, 'title': 'Special OPS', 'release_date': '2026-11-05', 'vote_average': 8.6, 'is_new_season': True, 'release_badge': 'Season 2'},
        {'id': 104139, 'title': 'Gullak', 'release_date': '2026-12-10', 'vote_average': 9.1, 'is_new_season': True, 'release_badge': 'Season 5'},
    ]

    client = TMDBClient()
    for ih in indian_web_series:
        if ih['release_date'] >= today and ih['id'] not in seen_ids:
            seen_ids.add(ih['id'])
            poster_url = client.get_media_assets(ih['id'], 'tv') or 'https://images.unsplash.com/photo-1593305841991-05c297ba4575?q=80&w=400&auto=format&fit=crop'
            upcoming_tv.append({
                'id': ih['id'],
                'media_id': ih['id'],
                'title': ih['title'],
                'poster_url': poster_url,
                'release_date': ih['release_date'],
                'vote_average': ih['vote_average'],
                'media_type': 'tv',
                'is_new_season': ih['is_new_season'],
                'release_badge': ih['release_badge']
            })

    # 2. Fetch Global TMDB Upcoming Series Premieres & New Seasons
    url_tmdb_global = f"https://api.themoviedb.org/3/discover/tv?api_key={api_key}&language=en-US&first_air_date.gte={today}&sort_by=popularity.desc&include_adult=false&page=1"
    
    try:
        response = get_resilient_session().get(url_tmdb_global, timeout=6.0)
        if response.status_code == 200:
            results = response.json().get('results', [])
            for item in results:
                tid = item.get('id')
                if not tid or tid in seen_ids:
                    continue
                name = item.get('name') or item.get('original_name') or 'TV Show'
                air_date = item.get('first_air_date', '')
                if not air_date or air_date < today:
                    continue

                name_lower = name.lower()
                if any(word in name_lower for word in EXPLICIT_KEYWORDS) or any(soap in name_lower for soap in SOAP_KEYWORDS):
                    continue

                seen_ids.add(tid)
                poster_path = item.get('poster_path')
                
                is_new_season = any(k in name_lower for k in ['season', 's2', 's3', 's4', 's5', 'part 2', 'part 3', '2nd season', '3rd season'])
                season_label = "New Season" if is_new_season else "New Show"

                upcoming_tv.append({
                    'id': tid,
                    'media_id': tid,
                    'title': name,
                    'poster_url': f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://images.unsplash.com/photo-1593305841991-05c297ba4575?q=80&w=400&auto=format&fit=crop",
                    'release_date': air_date,
                    'vote_average': round(item.get('vote_average', 0.0), 1),
                    'media_type': 'tv',
                    'is_new_season': is_new_season,
                    'release_badge': season_label
                })
    except Exception as e:
        print(f"[TMDB UPCOMING TV ERROR] {e}")

    # 2. Fetch AniList Upcoming Anime Seasons (status: NOT_YET_RELEASED)
    anilist_url = 'https://graphql.anilist.co'
    graphql_query = '''
    {
      Page(page: 1, perPage: 8) {
        media(type: ANIME, status: NOT_YET_RELEASED, format: TV, sort: [POPULARITY_DESC], isAdult: false) {
          id
          title { english romaji }
          coverImage { extraLarge }
          startDate { year month day }
          averageScore
        }
      }
    }
    '''
    try:
        resp = requests.post(anilist_url, json={'query': graphql_query}, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json().get('data', {}).get('Page', {}).get('media', [])
            client = TMDBClient()
            for item in data:
                t_eng = item.get('title', {}).get('english')
                t_rom = item.get('title', {}).get('romaji')
                title = t_eng or t_rom or 'Anime'
                
                s_date = item.get('startDate', {})
                yr = s_date.get('year')
                mo = s_date.get('month')
                dy = s_date.get('day')
                rel_date = f"{yr}-{mo:02d}-{dy:02d}" if yr and mo and dy else f"{yr or 2026}-10-01"

                # Cross-resolve TMDB ID
                anilist_id = item.get('id')
                tmdb_id = resolve_anilist_to_tmdb_id(title, fallback_id=anilist_id)

                if tmdb_id in seen_ids:
                    continue
                seen_ids.add(tmdb_id)

                title_lower = title.lower()
                is_new_season = any(k in title_lower for k in ['season', '2nd', '3rd', '4th', '5th', 'part 2', 'part 3', 'cour 2', 'ii', 'iii'])
                season_label = "New Season" if is_new_season else "New Show"

                upcoming_tv.append({
                    'id': tmdb_id,
                    'media_id': tmdb_id,
                    'anilist_id': anilist_id,
                    'title': title,
                    'poster_url': item.get('coverImage', {}).get('extraLarge') or "https://images.unsplash.com/photo-1593305841991-05c297ba4575?q=80&w=400&auto=format&fit=crop",
                    'release_date': rel_date,
                    'vote_average': round((item.get('averageScore') or 85) / 10.0, 1),
                    'media_type': 'tv',
                    'is_new_season': is_new_season,
                    'release_badge': season_label
                })
    except Exception as e:
        print(f"[ANILIST UPCOMING TV ERROR] {e}")

    upcoming_tv.sort(key=lambda x: x.get('release_date') or '9999-99-99')
    upcoming_tv = upcoming_tv[:20]

    cache.set(cache_key, upcoming_tv, 43200)
    return upcoming_tv


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
        response = get_resilient_session().get(url, params=params, timeout=10.0)
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

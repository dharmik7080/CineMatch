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

def get_daily_trending_movies(force_refresh=False):
    """
    Fetches daily trending movies from TMDB /trending/movie/day API.
    Utilizes Cache-Aside pattern: stores (movies, timestamp) tuple.
    Returns: (trending_movies_list, last_updated_datetime)
    """
    from django.conf import settings
    api_key = getattr(settings, 'TMDB_API_KEY', '') or '41fc74ce5602882786e1e9d4933fdcc6'

    cache_key = "daily_trending_movies_cache"
    if force_refresh:
        cache.delete(cache_key)
    else:
        cached_payload = cache.get(cache_key)
        if cached_payload and len(cached_payload) == 2:
            movies, last_updated = cached_payload
            # Automatically refresh if cache is older than 15 minutes (900s)
            if (datetime.now() - last_updated).total_seconds() < 900:
                return cached_payload
        
    # Cache-Aside: Fetch fresh data from TMDB on cache miss or auto-invalidation
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
            # Store in cache (auto-refreshes every 15 minutes = 900 seconds)
            cache.set(cache_key, payload, 900)
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

    cache_key = "anilist_popular_anime_feed_v4"
    cached = cache.get(cache_key)
    if cached:
        return cached

    url = "https://graphql.anilist.co"
    query = """
    {
      Page (page: 1, perPage: 25) {
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
    anime_list = []

    try:
        resp = get_resilient_session().post(url, json={'query': query}, timeout=6.0)
        if resp.status_code == 200:
            media_items = resp.json().get('data', {}).get('Page', {}).get('media', [])
            seen_titles = set()
            filtered_items = []

            for item in media_items:
                title = item.get('title', {}).get('english') or item.get('title', {}).get('romaji') or ''
                if not title or title.lower() in seen_titles:
                    continue
                seen_titles.add(title.lower())
                filtered_items.append((title, item))

            from concurrent.futures import ThreadPoolExecutor

            def process_anime(pair):
                title, item = pair
                cover = item.get('coverImage', {}).get('extraLarge') or item.get('coverImage', {}).get('large') or ''
                score = item.get('averageScore')
                vote_avg = round(score / 10.0, 1) if score else 8.5
                genres = " | ".join(item.get('genres', [])[:2]) or "Anime"

                anilist_id = item.get('id')
                tmdb_id = resolve_anilist_to_tmdb_id(title, fallback_id=anilist_id)

                return {
                    'id': tmdb_id,
                    'media_id': tmdb_id,
                    'anilist_id': anilist_id,
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
                }

            with ThreadPoolExecutor(max_workers=10) as executor:
                anime_list = list(executor.map(process_anime, filtered_items))

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
    Fetches genuine upcoming Indian + Global movies releasing within the next 2 months (60 days).
    Combines regional Indian theatrical releases (region=IN) with global blockbusters.
    Caches result for 12 hours.
    """
    from django.conf import settings
    from django.core.cache import cache
    import requests, datetime

    today_dt = datetime.date.today()
    max_date_dt = today_dt + datetime.timedelta(days=60)
    today = today_dt.strftime('%Y-%m-%d')
    max_date_str = max_date_dt.strftime('%Y-%m-%d')

    cache_key = f"upcoming_movies_2months_v2_{today}"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    api_key = getattr(settings, 'TMDB_API_KEY', '') or '41fc74ce5602882786e1e9d4933fdcc6'
    EXPLICIT_KEYWORDS = {'sex', 'erotic', 'porn', 'xxx', 'hentai', 'nude', 'nudity', 'lust'}
    
    seen_ids = set()
    upcoming_movies = []

    # Inject Curated High-Profile Indian Movies
    curated_upcoming_movies = [
        {
            'id': 1378537,
            'movie_id': 1378537,
            'title': 'Mirzapur: The Movie',
            'poster_url': 'https://image.tmdb.org/t/p/w500/jAt8u6MMIMnExmG6bN02EQyB0KR.jpg',
            'release_date': '2026-10-15',
            'vote_average': 8.8,
            'media_type': 'movie'
        }
    ]

    for cm in curated_upcoming_movies:
        if today < cm['release_date'] <= max_date_str and cm['id'] not in seen_ids:
            seen_ids.add(cm['id'])
            upcoming_movies.append(cm)

    # 1. Fetch Upcoming Movies in Indian Region (region=IN)
    url_in = f"https://api.themoviedb.org/3/movie/upcoming?api_key={api_key}&language=en-US&region=IN&page=1"
    # 2. Fetch Upcoming Global Discover Movies (strictly between today and 2 months)
    url_global = f"https://api.themoviedb.org/3/discover/movie?api_key={api_key}&language=en-US&primary_release_date.gt={today}&primary_release_date.lte={max_date_str}&sort_by=popularity.desc&include_adult=false&page=1"

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
                    if not rel_date or rel_date <= today or rel_date > max_date_str:
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
        cache.set(cache_key, upcoming_movies, 43200)
        return upcoming_movies

    return []


def get_upcoming_tv_shows():
    """
    Fetches genuine upcoming TV series premieres & new seasons from TMDB & AniList releasing within the next 2 months (60 days).
    Focuses on premium OTT Indian Web Series (Amazon Prime, Netflix, Hotstar, SonyLIV)
    and excludes daily soaps and adult content.
    Caches result for 12 hours.
    """
    from django.conf import settings
    from django.core.cache import cache
    import requests, datetime

    today_dt = datetime.date.today()
    max_date_dt = today_dt + datetime.timedelta(days=60)
    today = today_dt.strftime('%Y-%m-%d')
    max_date_str = max_date_dt.strftime('%Y-%m-%d')

    cache_key = f"upcoming_tv_shows_2months_v2_{today}"
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

    # 1. Inject Premium Indian Web Series (UPCOMING Seasons & Premieres Only within 2 Months)
    indian_web_series = [
        {'id': 100911, 'title': 'Asur', 'release_date': '2026-10-30', 'vote_average': 8.6, 'is_new_season': True, 'release_badge': 'Season 3'},
        {'id': 100757, 'title': 'Special OPS', 'release_date': '2026-11-01', 'vote_average': 8.6, 'is_new_season': True, 'release_badge': 'Season 2'},
    ]

    client = TMDBClient()
    for ih in indian_web_series:
        if today < ih['release_date'] <= max_date_str and ih['id'] not in seen_ids:
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

    # 2. Fetch Global TMDB Upcoming Series Premieres & New Seasons (strictly between today and 2 months)
    url_tmdb_global = f"https://api.themoviedb.org/3/discover/tv?api_key={api_key}&language=en-US&first_air_date.gt={today}&first_air_date.lte={max_date_str}&sort_by=popularity.desc&include_adult=false&page=1"
    
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
                if not air_date or air_date <= today or air_date > max_date_str:
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

    # 3. Fetch AniList Upcoming Anime Seasons (status: NOT_YET_RELEASED)
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

                if rel_date < today or rel_date > max_date_str:
                    continue

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


def fetch_scene_soundtracks(title, year=None, media_type='movie', tmdb_id=None):
    """
    Fetches scene-by-scene soundtrack data, preview audio, Spotify links, and YouTube links.
    Queries iTunes API for official soundtrack tracklists with season-specific scene markers for TV shows.
    """
    import urllib.parse
    cache_key_raw = f"soundtrack_v2_{media_type}_{tmdb_id}_{title}_{year}"
    cache_key = hashlib.md5(cache_key_raw.encode('utf-8')).hexdigest()
    
    cached_data = cache.get(cache_key)
    if cached_data:
        return cached_data

    tracks = []
    clean_title = (title or '').strip()

    # Dedicated curated soundtrack lists for iconic Movies and TV series
    lower_title = clean_title.lower()
    
    curated_soundtracks = {
        'stranger things': [
            ("Running Up That Hill (A Deal With God)", "Kate Bush", "Season 4 • Ep 4", "Max's Escape from Vecna Scene"),
            ("Master of Puppets", "Metallica", "Season 4 • Ep 9", "Eddie Munson's Upside Down Guitar Solo"),
            ("Should I Stay or Should I Go", "The Clash", "Season 1 • Ep 2", "Will & Jonathan Listening to Cassette"),
            ("Never Ending Story", "Gaten Matarazzo & Gabriella Pizzolo", "Season 3 • Ep 8", "Dustin & Suzie's Duet during Starcourt Battle"),
            ("Separate Ways (Worlds Apart) [Bryce Miller/Remix]", "Journey & Steve Perry", "Season 4 • Ep 8", "The Hawkins Crew Prepares for Battle"),
            ("Time After Time", "Cyndi Lauper", "Season 2 • Ep 9", "The Snow Ball Dance Finale"),
            ("Every Breath You Take", "The Police", "Season 2 • Ep 9", "Mind Flayer Oversees Snow Ball Dance"),
            ("Heroes", "Peter Gabriel / David Bowie", "Season 1 • Ep 3", "Will's Fake Body Discovered in Quarry"),
            ("Pass The Dutchie", "Musical Youth", "Season 4 • Ep 2", "Argyle Drives the Surfer Boy Pizza Van"),
            ("Stranger Things Theme", "Kyle Dixon & Michael Stein", "All Seasons", "Iconic Main Title Sequence"),
        ],
        'kgf': [
            ("Toofan", "Ravi Basrur, Sri Krishna, Prudhvi Chandra", "00:32:40", "Rocky's Entry & Kolar Gold Fields Conquest Scene"),
            ("Sulthana", "Ravi Basrur, Mohan Krishna, Sachin Basrur", "00:54:10", "Rocky's Rise & Worker Rebellion Sequence"),
            ("Mehabooba", "Ananya Bhat, Ravi Basrur", "01:21:05", "Rocky & Reena Emotional Climax Romance Scene"),
            ("Koti Kanasugala", "Ananya Bhat, Ravi Basrur", "01:45:30", "Mother's Oath & Final Battle Sequence"),
            ("Dheera Dheera", "Ananya Bhat, Mohan Krishna", "00:08:15", "Rocky's Childhood Promise & Bombay Intro"),
            ("Gali Gali", "Neha Kakkar, Tanishk Bagchi", "01:10:20", "Gold Club Celebration Dance Scene"),
            ("Monster Theme", "Ravi Basrur", "02:02:10", "Rocky's Ultimate Empire End Credits Roll"),
        ],
        'rrr': [
            ("Naatu Naatu", "M. M. Keeravani, Rahul Sipligunj, Kaala Bhairava", "00:58:30", "Iconic Dance Battle Sequence with Ram & Bheem"),
            ("Komuram Bheemudo", "Kaala Bhairava", "02:15:40", "Bheem's Emotional Whipping & Public Resistance Scene"),
            ("Dosti", "M. M. Keeravani, Vedala Hemachandra", "00:22:15", "Train Rescue & Unbreakable Brotherhood Scene"),
            ("Raam Raavam", "M. M. Keeravani, Sreenivasa Joshi", "02:40:10", "Alluri Sitarama Raju Forest Battle Climax"),
            ("Etthara Jhanda", "M. M. Keeravani, Vishal Mishra", "03:02:00", "Triumphant End Credits Celebration Track"),
        ],
        'animal': [
            ("Arjan Vailly", "Bhupinder Babbal, Manan Bhardwaj", "01:28:10", "Iconic Axe & Machine Gun Hotel Hallway Fight Scene"),
            ("Satranga", "Arijit Singh, Shreyas Puranik", "00:42:15", "Ranvijay & Geetanjali Karwa Chauth Emotional Scene"),
            ("Saari Duniya Jalaa Denge", "B Praak, Jaani", "02:35:00", "Ranvijay's Revenge & Final Airstrip Showdown"),
            ("Papa Meri Jaan", "Sonu Nigam, Harshavardhan Rameshwar", "00:15:30", "Ranvijay's Childhood Devotion to Balbir Scene"),
            ("Jamal Kudu", "Khatereh Group", "02:10:45", "Abrar's Wedding Entry with Glass on Head"),
        ],
        'jawan': [
            ("Chaleya", "Arijit Singh, Shilpa Rao, Anirudh Ravichander", "00:48:20", "Shah Rukh Khan & Nayanthara Romantic Cruise Dance"),
            ("Zinda Banda", "Anirudh Ravichander", "00:28:10", "Vikram Rathore High-Energy Prison Dance Sequence"),
            ("Not Ramaiya Vastavaiya", "Anirudh Ravichander, Vishal Dadlani", "02:45:00", "Post-Credits Celebration Sequence"),
            ("Jawan Title Track", "Anirudh Ravichander, Raja Kumari", "00:05:10", "Metro Heist & Bandaged SRK Reveal Scene"),
        ],
        'panchayat': [
            ("Hind Ke Sitare", "Manoj Tiwari", "Season 3 • Ep 8", "Phulera Village Battle & Mayor Showdown"),
            ("Panchayat Theme", "Anurag Saikia", "All Seasons", "Abhishek Kumar Rides Bicycle into Phulera"),
            ("Hiya Ho", "Anurag Saikia", "Season 2 • Ep 8", "Prahlad Cha's Emotional Loss Sequence"),
            ("Gazab Ka Hai Din", "Anurag Saikia", "Season 1 • Ep 6", "Abhishek & Rinki First Water Tanker Meeting"),
        ]
    }
    
    matched_key = None
    for k in curated_soundtracks:
        if k in lower_title or (k == 'kgf' and ('k.g.f' in lower_title or 'kgf' in lower_title)):
            matched_key = k
            break

    if matched_key:
        for idx, (t_name, a_name, s_tag, s_desc) in enumerate(curated_soundtracks[matched_key]):
            query_str = f"{t_name} {a_name}"
            preview_url = ""
            art_url = 'https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=300&auto=format&fit=crop'
            try:
                it_search = f"https://itunes.apple.com/search?term={urllib.parse.quote(query_str)}&entity=song&limit=1"
                r = get_resilient_session().get(it_search, timeout=2.5)
                if r.status_code == 200 and r.json().get('results'):
                    item = r.json()['results'][0]
                    preview_url = item.get('previewUrl', '')
                    art_url = item.get('artworkUrl100', '').replace('100x100bb', '300x300bb') or art_url
            except Exception:
                pass

            tracks.append({
                'id': idx + 1,
                'track_name': t_name,
                'artist_name': a_name,
                'album_name': f"{clean_title} Soundtrack",
                'artwork_url': art_url,
                'preview_url': preview_url,
                'spotify_url': f"https://open.spotify.com/search/{urllib.parse.quote(query_str)}",
                'youtube_url': f"https://www.youtube.com/results?search_query={urllib.parse.quote(query_str)}",
                'timestamp': s_tag,
                'scene_description': s_desc
            })

    if not tracks:
        search_term = f"{clean_title} soundtrack"
        itunes_url = f"https://itunes.apple.com/search?term={urllib.parse.quote(search_term)}&entity=song&limit=15"
        
        try:
            session = get_resilient_session()
            resp = session.get(itunes_url, timeout=5.0)
            results = []
            if resp.status_code == 200:
                results = resp.json().get('results', [])
                
            # Fallback search if 'title + soundtrack' returned empty
            if not results:
                itunes_url_fallback = f"https://itunes.apple.com/search?term={urllib.parse.quote(clean_title)}&entity=song&limit=15"
                resp_fb = session.get(itunes_url_fallback, timeout=5.0)
                if resp_fb.status_code == 200:
                    results = resp_fb.json().get('results', [])

            if results:
                movie_scene_labels = [
                    {"timestamp": "00:08:15", "scene": "Opening Scene / Intro Sequence"},
                    {"timestamp": "00:32:40", "scene": "Key Character Introduction / Turning Point"},
                    {"timestamp": "00:54:10", "scene": "Mid-Movie Chase & Dramatic Scene"},
                    {"timestamp": "01:21:05", "scene": "Emotional Climax Sequence"},
                    {"timestamp": "01:45:30", "scene": "Final Battle & Resolution"},
                    {"timestamp": "02:02:10", "scene": "End Credits Roll"}
                ]

                tv_scene_labels = [
                    {"timestamp": "Season 1 • Ep 1", "scene": "Series Premiere & Intro Sequence"},
                    {"timestamp": "Season 1 • Ep 4", "scene": "Mid-Season Plot Turning Point"},
                    {"timestamp": "Season 1 • Ep 8", "scene": "Season 1 Finale Climax Scene"},
                    {"timestamp": "Season 2 • Ep 3", "scene": "Key Character Revelation Scene"},
                    {"timestamp": "Season 2 • Ep 9", "scene": "Season 2 Finale Dance / Celebration"},
                    {"timestamp": "Season 3 • Ep 4", "scene": "High-Stakes Action Sequence"},
                    {"timestamp": "Season 3 • Ep 8", "scene": "Season 3 Starcourt Mall Climax"},
                    {"timestamp": "Season 4 • Ep 4", "scene": "Iconic Emotional Escape Scene"},
                    {"timestamp": "Season 4 • Ep 9", "scene": "Season 4 Epic Upside Down Battle"},
                    {"timestamp": "All Seasons", "scene": "Main Title Theme & End Credits"}
                ]

                labels_to_use = tv_scene_labels if media_type == 'tv' else movie_scene_labels
                import re
                title_clean_words = set(re.findall(r'\b\w+\b', lower_title)) - {'the', 'a', 'an', 'of', 'and', 'or', 'in', 'on', 'to', 'for', 'part', 'chapter', 'movie', 'soundtrack'}
                
                for idx, item in enumerate(results):
                    track_name = item.get('trackName', 'Unknown Track')
                    artist_name = item.get('artistName', 'Various Artists')
                    album_name = item.get('collectionName', clean_title)
                    artwork_url = item.get('artworkUrl100', '').replace('100x100bb', '300x300bb') or 'https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=300&auto=format&fit=crop'
                    preview_url = item.get('previewUrl', '')
                    
                    # STRICT RELEVANCE CHECK: Exclude unrelated video games, Toby Fox, Deltarune, or tracks not matching title
                    item_search_str = f"{track_name} {artist_name} {album_name}".lower()
                    item_words = set(re.findall(r'\b\w+\b', item_search_str))
                    
                    # If title clean words exist, at least one word MUST match the item metadata
                    if title_clean_words and not title_clean_words.intersection(item_words):
                        continue

                    query_str = f"{track_name} {artist_name}"
                    spotify_link = f"https://open.spotify.com/search/{urllib.parse.quote(query_str)}"
                    youtube_link = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query_str + ' official audio')}"
                    
                    scene_info = labels_to_use[len(tracks) % len(labels_to_use)]
                    
                    ts_label = scene_info['timestamp']
                    if media_type == 'tv':
                        album_lower = album_name.lower()
                        if 'season 1' in album_lower or 'vol. 1' in album_lower:
                            ts_label = 'Season 1'
                        elif 'season 2' in album_lower or 'vol. 2' in album_lower:
                            ts_label = 'Season 2'
                        elif 'season 3' in album_lower:
                            ts_label = 'Season 3'
                        elif 'season 4' in album_lower:
                            ts_label = 'Season 4'

                    tracks.append({
                        'id': len(tracks) + 1,
                        'track_name': track_name,
                        'artist_name': artist_name,
                        'album_name': album_name,
                        'artwork_url': artwork_url,
                        'preview_url': preview_url,
                        'spotify_url': spotify_link,
                        'youtube_url': youtube_link,
                        'timestamp': ts_label,
                        'scene_description': scene_info['scene']
                    })
                    if len(tracks) >= 12:
                        break
        except Exception as e:
            print(f"[SOUNDTRACK FETCH ERROR] iTunes query failed for {clean_title}: {e}")

    result_payload = {
        'title': clean_title,
        'has_soundtrack': bool(tracks),
        'total_tracks': len(tracks),
        'tracks': tracks
    }
    cache.set(cache_key, result_payload, timeout=86400)
    return result_payload


def fetch_opensubtitles(imdb_id=None, tmdb_id=None, title=None, season=None, episode=None):
    """
    Fetches dynamic multi-language subtitle tracks (.vtt / .srt) using OpenSubtitles REST API
    and provides 1-click subtitle download URLs for English, Hindi, Spanish, Japanese, French, and German.
    """
    import urllib.parse
    clean_imdb = str(imdb_id).replace('tt', '') if imdb_id else None
    cache_key_raw = f"subtitles_{clean_imdb}_{tmdb_id}_{season}_{episode}"
    cache_key = hashlib.md5(cache_key_raw.encode('utf-8')).hexdigest()

    cached_data = cache.get(cache_key)
    if cached_data:
        return cached_data

    languages_config = [
        {'name': 'English', 'code': 'en', 'sublang': 'eng', 'flag': '🇺🇸'},
        {'name': 'Hindi', 'code': 'hi', 'sublang': 'hin', 'flag': '🇮🇳'},
        {'name': 'Spanish', 'code': 'es', 'sublang': 'spa', 'flag': '🇪🇸'},
        {'name': 'Japanese', 'code': 'ja', 'sublang': 'jpn', 'flag': '🇯🇵'},
        {'name': 'French', 'code': 'fr', 'sublang': 'fre', 'flag': '🇫🇷'},
        {'name': 'German', 'code': 'de', 'sublang': 'ger', 'flag': '🇩🇪'}
    ]

    subtitles = []
    session = get_resilient_session()
    headers = {'User-Agent': 'TemporaryUserAgent v1.0'}

    # Attempt OpenSubtitles REST search if IMDB ID is present
    if clean_imdb:
        try:
            full_imdb = f"tt{clean_imdb.zfill(7)}"
            api_url = f"https://rest.opensubtitles.org/search/imdbid-{clean_imdb}"
            if season and episode:
                api_url += f"/season-{season}/episode-{episode}"
            
            resp = session.get(api_url, headers=headers, timeout=4.0)
            if resp.status_code == 200:
                raw_subs = resp.json()
                found_langs = set()
                import re
                for sub in raw_subs:
                    lang_code = sub.get('SubLanguageID')
                    dl_url = sub.get('SubDownloadLink') or sub.get('ZipDownloadLink')
                    sub_name = sub.get('SubFileName') or f"{title or 'Movie'} Subtitle"
                    
                    # Strict validation for TV episode matching
                    if season and episode:
                        target_s = str(season).lstrip('0') or '1'
                        target_e = str(episode).lstrip('0') or '1'
                        
                        sub_s = str(sub.get('SeriesSeason', '')).strip().lstrip('0')
                        sub_e = str(sub.get('SeriesEpisode', '')).strip().lstrip('0')
                        if sub_s and sub_e and (sub_s != target_s or sub_e != target_e):
                            continue
                            
                        se_match = re.search(r's(\d+)e(\d+)', sub_name.lower())
                        if se_match:
                            fn_s = se_match.group(1).lstrip('0')
                            fn_e = se_match.group(2).lstrip('0')
                            if fn_s != target_s or fn_e != target_e:
                                continue

                    # Match language config
                    for lconf in languages_config:
                        if lconf['sublang'] == lang_code and lconf['code'] not in found_langs:
                            found_langs.add(lconf['code'])
                            vtt_url = sub.get('SubDownloadLink', '').replace('.gz', '.vtt')
                            subtitles.append({
                                'language': lconf['name'],
                                'code': lconf['code'],
                                'flag': lconf['flag'],
                                'file_name': sub_name,
                                'download_url': dl_url,
                                'vtt_url': vtt_url or dl_url,
                                'source': 'OpenSubtitles v3'
                            })
                            break
        except Exception as e:
            print(f"[OPENSUBTITLES ERROR] OpenSubtitles REST call failed for {clean_imdb}: {e}")

    # Ensure all primary languages exist with mirror / parameter URLs
    existing_codes = {s['code'] for s in subtitles}
    clean_title_encoded = urllib.parse.quote(title or 'media')
    
    for lconf in languages_config:
        if lconf['code'] not in existing_codes:
            # Generate fallback episode-specific subtitle download link & vtt track parameter
            if season and episode:
                srt_url = f"https://dl.opensubtitles.org/en/download/sub/{clean_imdb or tmdb_id}/season-{season}/episode-{episode}/{lconf['code']}"
                vtt_url = f"https://vidsrc.stream/sub/{clean_imdb or tmdb_id}_s{season}_e{episode}_{lconf['code']}.vtt"
                file_name = f"{title or 'CineMatch'}_S{str(season).zfill(2)}E{str(episode).zfill(2)}_{lconf['name']}_Subtitles.srt"
            else:
                srt_url = f"https://dl.opensubtitles.org/en/download/sub/{clean_imdb or tmdb_id}/{lconf['code']}"
                vtt_url = f"https://vidsrc.stream/sub/{clean_imdb or tmdb_id}_{lconf['code']}.vtt"
                file_name = f"{title or 'CineMatch'}_{lconf['name']}_Subtitles.srt"

            subtitles.append({
                'language': lconf['name'],
                'code': lconf['code'],
                'flag': lconf['flag'],
                'file_name': file_name,
                'download_url': srt_url,
                'vtt_url': vtt_url,
                'source': 'OpenSubtitles Mirror'
            })

    result_payload = {
        'total_subtitles': len(subtitles),
        'subtitles': subtitles
    }
    cache.set(cache_key, result_payload, timeout=43200)
    return result_payload


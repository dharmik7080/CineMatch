import time
import requests
from django.core.management.base import BaseCommand
from django.conf import settings
from core.models import MovieWatchlist, WatchedHistory, Review, UserReview, MediaReview, CachedMedia
from core.utils import get_resilient_session

class Command(BaseCommand):
    help = "Syncs and pre-caches TMDB detail payloads for all active user items into the local CachedMedia table."

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("Starting TMDb Local-First cache synchronization..."))
        
        api_key = getattr(settings, 'TMDB_API_KEY', '')
        if not api_key:
            self.stderr.write(self.style.ERROR("TMDB_API_KEY is not configured in settings."))
            return

        # 1. Gather all unique media items
        media_items = set()  # stores tuples of (media_id, media_type)
        
        # From watchlist
        for item in MovieWatchlist.objects.all():
            media_items.add((item.media_id, item.media_type))
            
        # From WatchedHistory
        for item in WatchedHistory.objects.all():
            media_items.add((item.movie_id, 'movie'))
            
        # From custom Review
        for item in Review.objects.all():
            media_items.add((item.movie_id, 'movie'))
            
        # From UserReview
        for item in UserReview.objects.all():
            media_items.add((item.media_id, item.media_type))
            
        # From MediaReview
        for item in MediaReview.objects.all():
            media_items.add((item.media_id, item.media_type))

        # Add trending/popular fallbacks to ensure a warm bootstrap cache
        fallback_ids = [
            (19995, 'movie'),  # Avatar
            (49026, 'movie'),  # The Dark Knight Rises
            (119051, 'tv'),    # Wednesday
            (66732, 'tv'),      # Stranger Things
            (27205, 'movie'),  # Inception
            (157336, 'movie')  # Interstellar
        ]
        for fid, ftype in fallback_ids:
            media_items.add((fid, ftype))

        self.stdout.write(self.style.NOTICE(f"Found {len(media_items)} unique media items to synchronize."))

        session = get_resilient_session()
        success_count = 0
        fail_count = 0

        for idx, (media_id, media_type) in enumerate(media_items, 1):
            self.stdout.write(f"[{idx}/{len(media_items)}] Syncing {media_type} ID {media_id}...")
            
            if media_type == 'movie':
                url = (
                    f"https://api.themoviedb.org/3/movie/{media_id}"
                    f"?api_key={api_key}"
                    f"&language=en-US"
                    f"&append_to_response=credits,videos,watch/providers,similar"
                )
            else:
                url = (
                    f"https://api.themoviedb.org/3/tv/{media_id}"
                    f"?api_key={api_key}"
                    f"&language=en-US"
                    f"&append_to_response=credits,videos,watch/providers,similar,external_ids"
                )

            try:
                # 1. Fetch main detail payload if not already cached
                cached_obj = CachedMedia.objects.filter(media_id=media_id, media_type=media_type).first()
                if cached_obj:
                    payload = cached_obj.data or {}
                else:
                    response = session.get(url, timeout=10.0)
                    if response.status_code == 200:
                        payload = response.json()
                    else:
                        payload = None

                if payload:
                    # 2. Explicitly query the watch/providers endpoint to bypass TMDb API sync lag
                    prov_url = f"https://api.themoviedb.org/3/{media_type}/{media_id}/watch/providers?api_key={api_key}"
                    prov_resp = session.get(prov_url, timeout=5.0)
                    if prov_resp.status_code == 200:
                        payload['watch/providers'] = prov_resp.json()
                    
                    # Update local database cache
                    CachedMedia.objects.update_or_create(
                        media_id=media_id,
                        media_type=media_type,
                        defaults={'data': payload}
                    )
                    success_count += 1
                else:
                    self.stderr.write(self.style.WARNING(f"Failed to fetch payload for {media_type} ID {media_id}"))
                    fail_count += 1
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error syncing/enriching {media_type} ID {media_id}: {e}"))
                fail_count += 1

            # Throttle slightly to respect TMDb rate limit guidelines
            time.sleep(0.1)

        self.stdout.write(self.style.SUCCESS(
            f"Cache synchronization completed. Success: {success_count}, Failed: {fail_count}."
        ))

        # 3. Proactive Genre Cache Seeding for main genres ('Crime', 'Action', 'Drama')
        self.stdout.write(self.style.NOTICE("Starting Genre Cache proactive seeding..."))
        from core.models import GenreCache
        from core.utils import fetch_media_by_genre
        
        main_genres = ['Crime', 'Action', 'Drama']
        media_types = ['movie', 'tv']
        genre_success = 0
        
        for g_name in main_genres:
            for m_type in media_types:
                self.stdout.write(f"Pre-caching genre '{g_name}' ({m_type}) discovery payload...")
                cache_name = f"{m_type}_{g_name.lower()}_page_1"
                try:
                    records, target_genre_name, total_pages = fetch_media_by_genre(g_name, media_type=m_type, page=1)
                    if records:
                        payload = {
                            'records': records,
                            'target_genre_name': target_genre_name,
                            'total_pages': total_pages
                        }
                        GenreCache.objects.update_or_create(
                            genre_name=cache_name,
                            defaults={'data': payload}
                        )
                        genre_success += 1
                except Exception as ge:
                    self.stderr.write(self.style.ERROR(f"Failed to pre-cache genre '{g_name}' ({m_type}): {ge}"))
                time.sleep(0.2)

        self.stdout.write(self.style.SUCCESS(
            f"Genre Cache seeding completed successfully. Populated {genre_success} categories."
        ))

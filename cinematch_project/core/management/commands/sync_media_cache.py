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
                # Synchronous pre-fetching in background with robust 10s timeout
                response = session.get(url, timeout=10.0)
                if response.status_code == 200:
                    payload = response.json()
                    
                    # Update local database cache
                    CachedMedia.objects.update_or_create(
                        media_id=media_id,
                        media_type=media_type,
                        defaults={'data': payload}
                    )
                    success_count += 1
                else:
                    self.stderr.write(self.style.WARNING(f"API returned status {response.status_code} for {media_type} ID {media_id}"))
                    fail_count += 1
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error fetching {media_type} ID {media_id}: {e}"))
                fail_count += 1

            # Throttle slightly to respect TMDb rate limit guidelines
            time.sleep(0.1)

        self.stdout.write(self.style.SUCCESS(
            f"Cache synchronization completed. Success: {success_count}, Failed: {fail_count}."
        ))

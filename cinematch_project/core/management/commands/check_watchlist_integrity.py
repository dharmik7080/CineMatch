import os
import pickle
from django.core.management.base import BaseCommand
from django.conf import settings
from core.models import MovieWatchlist

class Command(BaseCommand):
    help = "Verifies if the media IDs currently in user watchlists exist in the vector similarity matrix index."

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("Starting Watchlist Vector Integrity Check..."))

        # 1. Load the current watchlist items from database
        movies_watchlist = list(MovieWatchlist.objects.filter(media_type='movie').values_list('media_id', flat=True).distinct())
        tv_watchlist = list(MovieWatchlist.objects.filter(media_type='tv').values_list('media_id', flat=True).distinct())

        self.stdout.write(f"Loaded {len(movies_watchlist)} movie watchlist IDs and {len(tv_watchlist)} TV show watchlist IDs from database.")

        # 2. Load movie index
        movie_dict_path = os.path.join(settings.BASE_DIR, 'movie_dict.pkl')
        movies_in_index = set()
        if os.path.exists(movie_dict_path):
            try:
                with open(movie_dict_path, 'rb') as f:
                    movie_data = pickle.load(f)
                if isinstance(movie_data, dict) and 'movie_id' in movie_data:
                    movies_in_index = set(movie_data['movie_id'].values())
                self.stdout.write(self.style.SUCCESS(f"Loaded movie index with {len(movies_in_index)} items."))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error loading movie_dict.pkl: {e}"))
        else:
            self.stderr.write(self.style.WARNING("movie_dict.pkl not found in base directory."))

        # 3. Load TV index
        tv_dict_path = os.path.join(settings.BASE_DIR, 'tv_dict.pkl')
        tv_in_index = set()
        if os.path.exists(tv_dict_path):
            try:
                with open(tv_dict_path, 'rb') as f:
                    tv_data = pickle.load(f)
                if isinstance(tv_data, dict) and 'id' in tv_data:
                    tv_in_index = set(tv_data['id'].values())
                self.stdout.write(self.style.SUCCESS(f"Loaded TV show index with {len(tv_in_index)} items."))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error loading tv_dict.pkl: {e}"))
        else:
            self.stderr.write(self.style.WARNING("tv_dict.pkl not found in base directory."))

        # 4. Check Movies Integrity
        movie_mismatches = [mid for mid in movies_watchlist if mid not in movies_in_index]
        if movie_mismatches:
            self.stderr.write(self.style.WARNING(
                f"WARNING: The following {len(movie_mismatches)} movie Watchlist ID(s) are MISSING from the vector similarity index: {movie_mismatches}"
            ))
        else:
            self.stdout.write(self.style.SUCCESS("INTEGRITY CHECK PASSED: All watchlist movies exist in the similarity index!"))

        # 5. Check TV Shows Integrity
        tv_mismatches = [tid for tid in tv_watchlist if tid not in tv_in_index]
        if tv_mismatches:
            self.stderr.write(self.style.WARNING(
                f"WARNING: The following {len(tv_mismatches)} TV Watchlist ID(s) are MISSING from the vector similarity index: {tv_mismatches}"
            ))
        else:
            self.stdout.write(self.style.SUCCESS("INTEGRITY CHECK PASSED: All watchlist TV shows exist in the similarity index!"))

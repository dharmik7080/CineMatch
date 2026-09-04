import datetime
from django.core.management.base import BaseCommand
from django.core.cache import cache
from core.utils import get_upcoming_movies, get_upcoming_tv_shows

class Command(BaseCommand):
    help = "Refreshes and pre-caches newly released movies, upcoming titles, and theatrical releases every morning."

    def handle(self, *args, **options):
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        self.stdout.write(self.style.NOTICE(f"[DAILY REFRESH] Running morning movie data update for {today}..."))

        # 1. Clear old cache keys for previous dates if needed
        self.stdout.write("Fetching fresh upcoming movies...")
        upcoming_movies = get_upcoming_movies()
        self.stdout.write(self.style.SUCCESS(f"Loaded {len(upcoming_movies)} upcoming movies."))

        self.stdout.write("Fetching fresh upcoming TV shows & new seasons...")
        upcoming_tv = get_upcoming_tv_shows()
        self.stdout.write(self.style.SUCCESS(f"Loaded {len(upcoming_tv)} upcoming TV shows."))

        self.stdout.write(self.style.SUCCESS(f"[DAILY REFRESH COMPLETE] Morning movie update completed for {today}!"))

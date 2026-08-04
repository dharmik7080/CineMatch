import os
import pickle
import numpy as np
from django.core.management.base import BaseCommand
from django.db import transaction
from core.models import CachedRecommendation
from django.conf import settings

class Command(BaseCommand):
    help = "Extracts top-15 cosine similarity matches from PKL matrix files and caches them in the database."

    def handle(self, *args, **options):
        self.stdout.write(self.style.WARNING("Starting recommendation database indexing..."))
        
        # 1. Resolve model directory paths
        models_dir = os.path.join(settings.BASE_DIR.parent, 'models')
        movie_dict_path = os.path.join(models_dir, 'movie_dict.pkl')
        movie_sim_path = os.path.join(models_dir, 'similarity.pkl')
        tv_dict_path = os.path.join(models_dir, 'tv_dict.pkl')
        tv_sim_path = os.path.join(models_dir, 'tv_similarity.pkl')

        # 2. Check existence of files
        for path in [movie_dict_path, movie_sim_path, tv_dict_path, tv_sim_path]:
            if not os.path.exists(path):
                self.stdout.write(self.style.ERROR(f"File not found: {path}"))
                return

        # 3. Clear existing cached recommendation data first to avoid duplicate indices
        self.stdout.write("--> Scrubbing legacy recommendation records...")
        CachedRecommendation.objects.all().delete()

        # ==========================================
        # SECTION A: PROCESS MOVIE RECOMMENDATIONS
        # ==========================================
        self.stdout.write("--> Processing Movies similarity matrix...")
        with open(movie_dict_path, 'rb') as f:
            movie_dict = pickle.load(f)
        with open(movie_sim_path, 'rb') as f:
            movie_similarity = pickle.load(f)

        movie_ids = list(movie_dict['movie_id'].values())
        movie_count = len(movie_ids)
        self.stdout.write(f"Found {movie_count} movies. Inserting records...")

        movie_recs_to_create = []
        
        for i, movie_id in enumerate(movie_ids):
            scores = movie_similarity[i]
            # argsort sorts in ascending order; [::-1] reverses it to descending.
            # We skip index 0 of the result because it is the movie itself (similarity = 1.0).
            top_indices = np.argsort(scores)[::-1][1:16]
            
            for idx in top_indices:
                if idx < len(movie_ids):
                    movie_recs_to_create.append(CachedRecommendation(
                        source_id=movie_id,
                        target_id=movie_ids[idx],
                        score=float(scores[idx]),
                        media_type='movie'
                    ))
            
            # Flush in chunks to prevent memory bloat during import
            if len(movie_recs_to_create) >= 5000:
                with transaction.atomic():
                    CachedRecommendation.objects.bulk_create(movie_recs_to_create)
                movie_recs_to_create = []

        if movie_recs_to_create:
            with transaction.atomic():
                CachedRecommendation.objects.bulk_create(movie_recs_to_create)

        self.stdout.write(self.style.SUCCESS("--> Movies similarity database import complete!"))

        # ==========================================
        # SECTION B: PROCESS TV SHOW RECOMMENDATIONS
        # ==========================================
        self.stdout.write("--> Processing TV Shows similarity matrix...")
        with open(tv_dict_path, 'rb') as f:
            tv_dict = pickle.load(f)
        with open(tv_sim_path, 'rb') as f:
            tv_similarity = pickle.load(f)

        tv_ids = list(tv_dict['id'].values())
        tv_count = len(tv_ids)
        self.stdout.write(f"Found {tv_count} TV shows. Inserting records...")

        tv_recs_to_create = []

        for i, tv_id in enumerate(tv_ids):
            scores = tv_similarity[i]
            top_indices = np.argsort(scores)[::-1][1:16]
            
            for idx in top_indices:
                if idx < len(tv_ids):
                    tv_recs_to_create.append(CachedRecommendation(
                        source_id=tv_id,
                        target_id=tv_ids[idx],
                        score=float(scores[idx]),
                        media_type='tv'
                    ))
            
            # Flush in chunks
            if len(tv_recs_to_create) >= 5000:
                with transaction.atomic():
                    CachedRecommendation.objects.bulk_create(tv_recs_to_create)
                tv_recs_to_create = []

        if tv_recs_to_create:
            with transaction.atomic():
                CachedRecommendation.objects.bulk_create(tv_recs_to_create)

        self.stdout.write(self.style.SUCCESS("--> TV Shows similarity database import complete!"))
        
        total_records = CachedRecommendation.objects.count()
        self.stdout.write(self.style.SUCCESS(f"=== DB Cache Ingestion Complete! Inserted {total_records} similarity indices. ==="))

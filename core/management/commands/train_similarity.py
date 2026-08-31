from django.core.management.base import BaseCommand
from core.models import CachedMedia, CachedRecommendation
from django.conf import settings
import os
import pickle
import numpy as np
import re

class Command(BaseCommand):
    help = "Train and update similarity recommendations using both pre-existing PKL vectors and new CachedMedia items."

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'<[^>]*>', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return re.sub(r'\s+', ' ', text).lower().strip()

    def handle(self, *args, **options):
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            self.stdout.write(self.style.ERROR(
                "scikit-learn is not installed in this environment. "
                "Please run: pip install scikit-learn"
            ))
            return

        models_dir = os.path.join(settings.BASE_DIR, 'models')
        movie_dict_path = os.path.join(models_dir, 'movie_dict.pkl')
        movie_sim_path = os.path.join(models_dir, 'similarity.pkl')
        tv_dict_path = os.path.join(models_dir, 'tv_dict.pkl')
        tv_sim_path = os.path.join(models_dir, 'tv_similarity.pkl')

        # ----------------------------------------------------
        # PROCESS MOVIES (Precomputed + CachedMedia)
        # ----------------------------------------------------
        self.stdout.write("Loading pre-computed movie dataset...")
        
        movie_ids = []
        movie_titles = []
        movie_tags = []
        
        if os.path.exists(movie_dict_path):
            with open(movie_dict_path, 'rb') as f:
                movie_dict = pickle.load(f)
            movie_ids = list(movie_dict['movie_id'].values())
            movie_titles = list(movie_dict['title'].values())
            movie_tags = list(movie_dict['tags'].values())
        else:
            self.stdout.write(self.style.WARNING("movie_dict.pkl not found. Relying solely on database."))

        self.stdout.write("Fetching cached movie database...")
        db_movies = list(CachedMedia.objects.filter(media_type='movie'))
        new_movies_count = 0
        
        for item in db_movies:
            if item.media_id not in movie_ids:
                data = item.data or {}
                title = data.get('title') or data.get('name') or ''
                
                genres = " ".join([g.get('name', '') for g in data.get('genres', []) if g.get('name')])
                overview = data.get('overview') or ''
                tagline = data.get('tagline') or ''
                
                credits_data = data.get('credits', {})
                cast = " ".join([c.get('name', '') for c in credits_data.get('cast', [])[:5] if c.get('name')])
                director = " ".join([m.get('name', '') for m in credits_data.get('crew', []) if m.get('job') == 'Director'])
                
                raw_tags = f"{title} {genres} {overview} {tagline} {cast} {director}"
                cleaned_tags = self.clean_text(raw_tags)
                
                movie_ids.append(item.media_id)
                movie_titles.append(title)
                movie_tags.append(cleaned_tags)
                new_movies_count += 1

        total_movies = len(movie_ids)
        self.stdout.write(f"Total Movies: {total_movies} ({new_movies_count} new database items).")

        movie_recs = []
        if total_movies >= 2:
            self.stdout.write("Vectorizing movie tags...")
            cv = CountVectorizer(max_features=5000, stop_words='english')
            movie_vectors = cv.fit_transform(movie_tags).toarray()
            
            self.stdout.write("Computing pairwise Movie similarity...")
            movie_similarity = cosine_similarity(movie_vectors)
            
            for i in range(total_movies):
                source_id = movie_ids[i]
                scores = movie_similarity[i]
                top_indices = np.argsort(scores)[::-1][1:16]
                
                for idx in top_indices:
                    if idx < len(movie_ids):
                        score = scores[idx]
                        if score >= 0.05:
                            movie_recs.append(CachedRecommendation(
                                source_id=source_id,
                                target_id=movie_ids[idx],
                                score=float(round(score, 4)),
                                media_type='movie'
                            ))

        # ----------------------------------------------------
        # PROCESS TV SHOWS (Precomputed + CachedMedia)
        # ----------------------------------------------------
        self.stdout.write("Loading pre-computed TV dataset...")
        
        tv_ids = []
        tv_titles = []
        tv_tags = []
        
        if os.path.exists(tv_dict_path):
            with open(tv_dict_path, 'rb') as f:
                tv_dict = pickle.load(f)
            tv_ids = list(tv_dict['id'].values())
            tv_titles = list(tv_dict['title'].values())
            tv_tags = list(tv_dict['tags'].values())
        else:
            self.stdout.write(self.style.WARNING("tv_dict.pkl not found. Relying solely on database."))

        self.stdout.write("Fetching cached TV database...")
        db_tv = list(CachedMedia.objects.filter(media_type='tv'))
        new_tv_count = 0
        
        for item in db_tv:
            if item.media_id not in tv_ids:
                data = item.data or {}
                title = data.get('title') or data.get('name') or ''
                
                genres = " ".join([g.get('name', '') for g in data.get('genres', []) if g.get('name')])
                overview = data.get('overview') or ''
                tagline = data.get('tagline') or ''
                
                credits_data = data.get('credits', {})
                cast = " ".join([c.get('name', '') for c in credits_data.get('cast', [])[:5] if c.get('name')])
                director = " ".join([m.get('name', '') for m in credits_data.get('crew', []) if m.get('job') == 'Director'])
                
                raw_tags = f"{title} {genres} {overview} {tagline} {cast} {director}"
                cleaned_tags = self.clean_text(raw_tags)
                
                tv_ids.append(item.media_id)
                tv_titles.append(title)
                tv_tags.append(cleaned_tags)
                new_tv_count += 1

        total_tv = len(tv_ids)
        self.stdout.write(f"Total TV Shows: {total_tv} ({new_tv_count} new database items).")

        tv_recs = []
        if total_tv >= 2:
            self.stdout.write("Vectorizing TV tags...")
            cv = CountVectorizer(max_features=5000, stop_words='english')
            tv_vectors = cv.fit_transform(tv_tags).toarray()
            
            self.stdout.write("Computing pairwise TV similarity...")
            tv_similarity = cosine_similarity(tv_vectors)
            
            for i in range(total_tv):
                source_id = tv_ids[i]
                scores = tv_similarity[i]
                top_indices = np.argsort(scores)[::-1][1:16]
                
                for idx in top_indices:
                    if idx < len(tv_ids):
                        score = scores[idx]
                        if score >= 0.05:
                            tv_recs.append(CachedRecommendation(
                                source_id=source_id,
                                target_id=tv_ids[idx],
                                score=float(round(score, 4)),
                                media_type='tv'
                            ))

        # ----------------------------------------------------
        # DB CACHE INGESTION (Wipe and bulk write)
        # ----------------------------------------------------
        self.stdout.write("Writing all recommendation indices to database cache...")
        try:
            # Wipe recommendations
            CachedRecommendation.objects.all().delete()
            
            # Merge lists
            total_recs = movie_recs + tv_recs
            
            # Bulk create
            CachedRecommendation.objects.bulk_create(total_recs, batch_size=2000)
            
            self.stdout.write(self.style.SUCCESS(
                f"Successfully mapped recommendations database. "
                f"Generated {len(total_recs)} similarity indices."
            ))
        except Exception as db_err:
            self.stdout.write(self.style.ERROR(f"Database sync failed: {db_err}"))

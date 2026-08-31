from django.core.management.base import BaseCommand
from core.models import CachedMedia, CachedRecommendation
import re

class Command(BaseCommand):
    help = "Train and update the local content similarity recommendation engine using CachedMedia."

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        # Remove HTML tags
        text = re.sub(r'<[^>]*>', ' ', text)
        # Keep letters, numbers, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Lowercase and strip extra space
        return re.sub(r'\s+', ' ', text).lower().strip()

    def handle(self, *args, **options):
        # Import sklearn dependencies inside handle to avoid import crashes at Django startup
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            self.stdout.write(self.style.ERROR(
                "scikit-learn is not installed in this environment. "
                "Please run: pip install scikit-learn"
            ))
            return

        self.stdout.write("Fetching all locally cached media items...")
        media_items = list(CachedMedia.objects.all())
        total_items = len(media_items)
        
        if total_items < 2:
            self.stdout.write(self.style.WARNING(
                f"Insufficient media items in cache ({total_items}). Need at least 2 to train similarity."
            ))
            return

        self.stdout.write(f"Pre-processing tags for {total_items} items...")
        
        media_keys = []
        tags_list = []
        
        for item in media_items:
            data = item.data or {}
            title = data.get('title') or data.get('name') or ''
            
            # Genres
            genres_list = data.get('genres', [])
            genres = " ".join([g.get('name', '') for g in genres_list if g.get('name')])
            
            overview = data.get('overview') or ''
            tagline = data.get('tagline') or ''
            
            # Cast & Crew
            credits_data = data.get('credits', {})
            cast_list = credits_data.get('cast', [])[:5]
            cast = " ".join([c.get('name', '') for c in cast_list if c.get('name')])
            
            crew_list = credits_data.get('crew', [])
            directors = [m.get('name', '') for m in crew_list if m.get('job') == 'Director']
            director = " ".join(directors)
            
            # Combine raw elements
            raw_tags = f"{title} {genres} {overview} {tagline} {cast} {director}"
            cleaned_tags = self.clean_text(raw_tags)
            
            media_keys.append((item.media_id, item.media_type))
            tags_list.append(cleaned_tags)

        self.stdout.write("Vectorizing tags with CountVectorizer...")
        cv = CountVectorizer(max_features=5000, stop_words='english')
        try:
            vectors = cv.fit_transform(tags_list).toarray()
        except Exception as ve:
            self.stdout.write(self.style.ERROR(f"Vectorization failed: {ve}"))
            return

        self.stdout.write("Computing pairwise Cosine Similarity matrix...")
        similarity_matrix = cosine_similarity(vectors)

        self.stdout.write("Generating recommendation records...")
        new_recommendations = []
        
        for i in range(total_items):
            source_id, media_type = media_keys[i]
            
            # Get similarity scores relative to all other items
            scores = list(enumerate(similarity_matrix[i]))
            
            # Sort by score descending (excluding itself)
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            
            count = 0
            for target_idx, score in scores:
                if target_idx == i:
                    continue  # Exclude self
                if score < 0.05:
                    break  # Cutoff low-scoring similarities
                    
                target_id, target_type = media_keys[target_idx]
                
                new_recommendations.append(CachedRecommendation(
                    source_id=source_id,
                    target_id=target_id,
                    score=float(round(score, 4)),
                    media_type=media_type
                ))
                
                count += 1
                if count >= 10:  # Top 10 matches
                    break

        self.stdout.write("Updating database cache...")
        try:
            # Wipe old recommendations
            CachedRecommendation.objects.all().delete()
            # Bulk create to make database writing incredibly fast
            CachedRecommendation.objects.bulk_create(new_recommendations, batch_size=1000)
            self.stdout.write(self.style.SUCCESS(
                f"Successfully trained content-similarity models on {total_items} items. "
                f"Generated {len(new_recommendations)} recommendation link mappings."
            ))
        except Exception as db_err:
            self.stdout.write(self.style.ERROR(f"Database sync failed: {db_err}"))

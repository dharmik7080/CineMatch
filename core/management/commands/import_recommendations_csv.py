import os
import csv
from django.core.management.base import BaseCommand
from django.db import transaction
from django.conf import settings
from core.models import CachedRecommendation

class Command(BaseCommand):
    help = "Ingests cosine similarity records line-by-line from a CSV file to minimize memory usage."

    def handle(self, *args, **options):
        self.stdout.write(self.style.WARNING("Starting memory-efficient recommendation database ingestion..."))
        
        csv_path = os.path.join(settings.BASE_DIR, 'recommendations.csv')
        if not os.path.exists(csv_path):
            self.stdout.write(self.style.ERROR(f"File not found: {csv_path}"))
            return

        self.stdout.write("--> Scrubbing legacy recommendation records...")
        CachedRecommendation.objects.all().delete()

        self.stdout.write("--> Ingesting similarity records...")
        
        recs_to_create = []
        batch_size = 5000
        count = 0

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                source_id, target_id, score, media_type = row
                
                recs_to_create.append(CachedRecommendation(
                    source_id=int(source_id),
                    target_id=int(target_id),
                    score=float(score),
                    media_type=media_type
                ))
                count += 1

                if len(recs_to_create) >= batch_size:
                    with transaction.atomic():
                        CachedRecommendation.objects.bulk_create(recs_to_create)
                    recs_to_create = []
                    self.stdout.write(f"--> Ingested {count} records...")

            if recs_to_create:
                with transaction.atomic():
                    CachedRecommendation.objects.bulk_create(recs_to_create)

        self.stdout.write(self.style.SUCCESS(f"=== DB Ingestion Complete! Successfully loaded {count} similarity records. ==="))

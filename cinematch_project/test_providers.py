import os
import django
import requests

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cinematch_project.settings')
django.setup()

from django.conf import settings

def get_all_providers():
    url = "https://api.themoviedb.org/3/watch/providers/movie"
    params = {'api_key': settings.TMDB_API_KEY, 'watch_region': 'IN'}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        providers = response.json().get('results', [])
        print(f"{'ID':<10} | {'Provider Name'}")
        print("-" * 30)
        for p in providers:
            print(f"{p['provider_id']:<10} | {p['provider_name']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    get_all_providers()
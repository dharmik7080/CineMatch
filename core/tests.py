from django.test import TestCase
from django.urls import reverse

class UniversalSearchTests(TestCase):
    def test_universal_search_empty_query(self):
        response = self.client.get(reverse('universal_search'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['movies'], [])
        self.assertEqual(data['tv_shows'], [])
        self.assertEqual(data['people'], [])

    def test_universal_search_with_query(self):
        response = self.client.get(reverse('universal_search'), {'q': 'Inception'})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('movies', data)
        self.assertIn('tv_shows', data)
        self.assertIn('people', data)


class SearchResultsPageTests(TestCase):
    def test_search_results_empty_query(self):
        response = self.client.get(reverse('search_results'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No results found")

    def test_search_results_with_query(self):
        response = self.client.get(reverse('search_results'), {'q': 'Inception'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Search Results")
        self.assertContains(response, "Inception")


from django.contrib.auth.models import User
from core.models import RecommendationFeedback
from unittest.mock import patch
import numpy as np

class ExploreViewsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='password')
        self.client.login(username='testuser', password='password')

    def test_explore_movies_page(self):
        response = self.client.get(reverse('explore_movies'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Explore Movie Library")

    def test_explore_tv_page(self):
        response = self.client.get(reverse('explore_tv'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Explore TV Series Library")


class RecommendationFeedbackTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='recommendation-user', password='password')
        self.client.login(username='recommendation-user', password='password')

    def test_not_interested_hides_a_title_from_future_recommendations(self):
        response = self.client.post(
            reverse('mark_not_interested'),
            {'media_id': 157336, 'media_type': 'movie'},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['success'])
        self.assertTrue(
            RecommendationFeedback.objects.filter(
                user=self.user, media_id=157336, media_type='movie'
            ).exists()
        )

    @patch('core.views.get_cached_poster', return_value='https://example.test/poster.jpg')
    @patch('core.views.TMDBClient')
    def test_recommender_excludes_not_interested_titles(self, mock_client, _mock_poster):
        from core import views

        MovieWatchlist = views.MovieWatchlist
        MovieWatchlist.objects.create(user=self.user, media_id=1, media_type='movie')
        RecommendationFeedback.objects.create(user=self.user, media_id=2, media_type='movie')
        mock_client.return_value.get_streaming_or_theatre_links.return_value = ''
        original_dict, original_similarity = views.MOVIE_DICT, views.MOVIE_SIMILARITY
        try:
            views.MOVIE_DICT = {
                'movie_id': {0: 1, 1: 2, 2: 3},
                'title': {0: 'Seed', 1: 'Hidden', 2: 'Visible'},
                'tags': {0: 'space', 1: 'space drama', 2: 'space adventure'},
            }
            views.MOVIE_SIMILARITY = np.array([
                [1.0, 0.95, 0.90],
                [0.95, 1.0, 0.60],
                [0.90, 0.60, 1.0],
            ])
            results = views.get_recommendations([1], 'movie', user=self.user)
        finally:
            views.MOVIE_DICT, views.MOVIE_SIMILARITY = original_dict, original_similarity

        self.assertEqual([item['movie_id'] for item in results], [3])
        self.assertEqual(results[0]['recommendation_reason'], 'Because you saved Seed')

    @patch('core.views.get_cached_poster', return_value='https://example.test/poster.jpg')
    @patch('core.views.TMDBClient')
    def test_recommender_limits_repeated_franchises(self, mock_client, _mock_poster):
        from core import views

        views.MovieWatchlist.objects.create(user=self.user, media_id=1, media_type='movie')
        mock_client.return_value.get_streaming_or_theatre_links.return_value = ''
        original_dict, original_similarity = views.MOVIE_DICT, views.MOVIE_SIMILARITY
        try:
            views.MOVIE_DICT = {
                'movie_id': {0: 1, 1: 2, 2: 3, 3: 4},
                'title': {
                    0: 'Seed Film', 1: 'Harry Potter and the Sorcerer Stone',
                    2: 'Harry Potter and the Chamber of Secrets', 3: 'Arrival',
                },
                'tags': {0: 'magic', 1: 'magic wizard', 2: 'magic wizard', 3: 'science fiction'},
            }
            views.MOVIE_SIMILARITY = np.array([
                [1.0, 0.98, 0.97, 0.60],
                [0.98, 1.0, 0.99, 0.20],
                [0.97, 0.99, 1.0, 0.20],
                [0.60, 0.20, 0.20, 1.0],
            ])
            results = views.get_recommendations([1], 'movie', user=self.user)
        finally:
            views.MOVIE_DICT, views.MOVIE_SIMILARITY = original_dict, original_similarity

        titles = [item['title'] for item in results]
        self.assertEqual(sum(title.startswith('Harry Potter') for title in titles), 1)
        self.assertIn('Arrival', titles)


class AnalyticsDashboardTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser2', password='password')
        self.client.login(username='testuser2', password='password')

    def test_analytics_dashboard_view_renders(self):
        response = self.client.get(reverse('analytics_dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('heatmap_img', response.context)
        self.assertIn('plotly_div', response.context)
        self.assertIn('network_img', response.context)
        # Verify that the Seaborn heatmap and Plotly scatter are not empty
        self.assertTrue(len(response.context['heatmap_img']) > 0)
        self.assertTrue(len(response.context['plotly_div']) > 0)
        self.assertTrue(len(response.context['network_img']) > 0)


class NormalizerTests(TestCase):
    def test_normalize_name(self):
        from core.utils import normalize_name
        
        # Test standard known services
        self.assertEqual(normalize_name("hotstar"), "disney+")
        self.assertEqual(normalize_name("JioHotstar"), "disney+")
        self.assertEqual(normalize_name("jio hotstar"), "disney+")
        self.assertEqual(normalize_name("Disney+ Hotstar"), "disney+")
        self.assertEqual(normalize_name("Prime Video"), "prime video")
        self.assertEqual(normalize_name("netflix"), "netflix")
        
        # Test unknown/missing service (defaults to original name)
        self.assertEqual(normalize_name("SomeUnknownService"), "SomeUnknownService")
        self.assertEqual(normalize_name(""), "")


class GenreViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser_genre', password='password')
        self.client.login(username='testuser_genre', password='password')

    def test_movies_by_genre_view(self):
        response = self.client.get(reverse('movies_by_genre', kwargs={'genre_name': 'Sci-Fi'}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Genre: Sci-Fi")

    def test_tv_shows_by_genre_view(self):
        # Should normalize Sci-Fi -> Sci-Fi & Fantasy for TV shows
        response = self.client.get(reverse('tv_shows_by_genre', kwargs={'genre_name': 'Sci-Fi'}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Genre: Sci-Fi &amp; Fantasy")

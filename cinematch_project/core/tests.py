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




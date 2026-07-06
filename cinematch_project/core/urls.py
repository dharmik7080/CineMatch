"""
CineMatch Project - App URLs Configuration
Syllabus Reference: Unit 8: Django routing & MVT architecture
"""

from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    # Custom root redirect to feed
    path('', views.for_you_feed, name='home'),

    # User Registration (Custom View)
    path('register/', views.register_user, name='register'),
    
    # User Login & Logout (Django's inbuilt auth views)
    path('login/', auth_views.LoginView.as_view(template_name='core/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    
    # "For You" Feed dashboard
    path('feed/', views.for_you_feed, name='for_you_feed'),
    
    # Explore Grids
    path('movies/', views.explore_movies, name='explore_movies'),
    path('tv-shows/', views.explore_tv, name='explore_tv'),
    
    # Analytics Panel
    path('analytics/', views.analytics_dashboard, name='analytics_dashboard'),
    
    # Watchlist CRUD endpoints (AJAX compatible)
    path('watchlist/add/', views.watchlist_add, name='watchlist_add'),
    path('watchlist/delete/', views.watchlist_delete, name='watchlist_delete'),

    # Movie Detail Hub (Unit 7: REST API deep-fetch with append_to_response)
    path('movies/<int:movie_id>/', views.movie_detail_view, name='movie_detail'),

    # TV Show Detail Hub (Unit 7: REST API compound request — TV variant)
    path('tv/<int:series_id>/', views.tv_detail_view, name='tv_show_detail'),

    # Watchlist
    path('watchlist/', views.watchlist_hub_view, name='watchlist_hub'),

    # ── 💎 ADDED: Media Review CRUD Ingestion Channels ──
    path('review/add/<str:media_type>/<int:media_id>/', views.add_media_review, name='add_media_review'),
    path('review/update/<int:review_id>/', views.update_media_review, name='update_media_review'),
    path('review/delete/<int:review_id>/', views.delete_media_review, name='delete_media_review'),

    # Unified Person Profile (Actor & Director) Route
    path('person/<int:person_id>/', views.person_profile, name='person_profile'),

    # Custom User Review Submission Route
    path('review/submit/<int:movie_id>/', views.submit_review, name='submit_review'),

    # Universal Search AJAX Channel
    path('universal-search/', views.universal_search, name='universal_search'),

    # Full Search Results Page Route
    path('search/', views.search_results_view, name='search_results'),
]
"""
CineMatch Project - App URLs Configuration
Syllabus Reference: Unit 8: Django routing & MVT architecture
"""

from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    # Intelligent Root Routing
    path('', views.home_redirect, name='home'),

    # User Registration / Signup
    path('register/', views.signup_view, name='register'),
    path('signup/', views.signup_view, name='signup'),
    
    # Custom User Login & Logout
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # "For You" Feed dashboard
    path('feed/', views.for_you_feed, name='for_you_feed'),
    
    # Explore Grids
    path('movies/', views.explore_movies, name='explore_movies'),
    path('movies/random/', views.random_movie_view, name='random_movie'),
    path('movies/genre/<str:genre_name>/', views.movies_by_genre_view, name='movies_by_genre'),
    path('tv-shows/', views.explore_tv, name='explore_tv'),
    path('tv-shows/random/', views.random_tv_show_view, name='random_tv'),
    path('tv/genre/<str:genre_name>/', views.tv_shows_by_genre_view, name='tv_shows_by_genre'),
    path('anime/', views.explore_anime_view, name='explore_anime'),
    
    # Analytics Panel
    path('analytics/', views.analytics_dashboard, name='analytics_dashboard'),
    
    # Watchlist CRUD endpoints (AJAX compatible)
    path('watchlist/add/', views.watchlist_add, name='watchlist_add'),
    path('watchlist/delete/', views.watchlist_delete, name='watchlist_delete'),
    path('recommendations/not-interested/', views.mark_not_interested, name='mark_not_interested'),

    # Movie Detail Hub (Unit 7: REST API deep-fetch with append_to_response)
    path('movies/<int:movie_id>/', views.movie_detail_view, name='movie_detail'),

    # TV Show Detail Hub (Unit 7: REST API compound request — TV variant)
    path('tv/<int:series_id>/', views.tv_detail_view, name='tv_show_detail'),
    path('tv/<int:series_id>/season/<int:season_number>/', views.tv_season_ajax, name='tv_season_ajax'),

    # Watchlist
    path('watchlist/', views.watchlist_hub_view, name='watchlist_hub'),
    path('watchlist/sync-history/', views.sync_continue_watching_view, name='sync_continue_watching'),

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

    # Collaborative Watch Groups (Watch Party Feature)
    path('groups/', views.group_list_view, name='group_list'),
    path('groups/create/', views.create_group_view, name='create_watch_group'),
    path('groups/join/<uuid:invite_code>/', views.join_group_view, name='join_watch_group'),
    path('groups/<uuid:group_code>/', views.group_dashboard_view, name='group_dashboard'),
    path('groups/<uuid:group_code>/add-item/', views.add_group_item_view, name='add_group_item'),
    path('groups/<uuid:group_code>/remove-item/<int:item_id>/', views.remove_group_item_view, name='remove_group_item'),
    path('groups/<uuid:group_code>/message/send/', views.send_group_message_view, name='send_group_message'),

    # Notifications System patterns
    path('notifications/', views.notifications_list_view, name='notifications_list'),
    path('notifications/read/<int:notification_id>/', views.mark_notification_read, name='mark_notification_read'),
    path('notifications/mark-all-read/', views.mark_all_notifications_read, name='mark_all_notifications_read'),

    # Temporary route to trigger fixture loading on Render free tier
    path('temp-load-fixtures-route/', views.temp_load_fixture, name='temp_load_fixture'),
]

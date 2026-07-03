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
]

# from django.contrib import admin
# from .models import Movie, Watchlist, WatchedHistory # Import your models here

# # Register your models here
# admin.site.register(Movie)
# admin.site.register(Watchlist)
# admin.site.register(WatchedHistory)

from django.contrib import admin
from .models import (
    UserProfile, 
    MovieWatchlist, 
    UserReview, 
    MediaReview, 
    Review, 
    WatchedHistory
)

# Admin configuration for WatchedHistory to make analytics data readable
@admin.register(WatchedHistory)
class WatchedHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'movie_title', 'duration', 'rating', 'watched_at')
    list_filter = ('watched_at', 'rating')
    search_fields = ('movie_title', 'user__username')

# Admin configuration for Review to see ratings at a glance
@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ('user', 'movie_title', 'rating', 'created_at')
    list_filter = ('rating', 'created_at')
    search_fields = ('movie_title', 'user__username')

# Admin configuration for MediaReview (useful for Sentiment Analysis tracking)
@admin.register(MediaReview)
class MediaReviewAdmin(admin.ModelAdmin):
    list_display = ('user', 'media_id', 'media_type', 'sentiment_score', 'updated_at')
    list_filter = ('media_type', 'sentiment_score')

# Registering the remaining models
@admin.register(MovieWatchlist)
class MovieWatchlistAdmin(admin.ModelAdmin):
    list_display = ('user', 'media_id', 'media_type', 'added_at')
    list_filter = ('media_type',)

admin.site.register(UserProfile)
admin.site.register(UserReview)
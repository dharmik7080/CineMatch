"""
CineMatch Project - Phase 2: Relational Database Design & Django Integration
Sub-Phase 2.1: Django Models and Relational DB Schema Implementation

Syllabus Reference:
- Unit 8: Django Framework & MVT Architecture (Model-View-Template)
- Unit 9: Django Models, Users, and Relational Database Schema Design (ORM Fields, Relationships)
"""

from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid

# ======================================================================
# Model 1: UserProfile
# Relational Concept: One-to-One Relationship (1:1)
# Syllabus Topic: Django Users & Schema Relationships (Unit 9)
# ======================================================================
class UserProfile(models.Model):
    """
    Extends Django's native auth User model to store domain-specific user details.
    Uses models.OneToOneField to establish a strict 1:1 mapping with the User table.
    """
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,      # Referential Integrity: deletes profile when User is deleted
        related_name='profile'         # Reverse lookup name (e.g. user.profile)
    )
    bio = models.TextField(blank=True, max_length=500, help_text="Short biography of the user.")
    favorite_genre = models.CharField(blank=True, max_length=100, help_text="User's preferred movie/TV genre.")

    class Meta:
        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"

    def __name__(self):
        return f"{self.user.username}'s Profile"

    def __str__(self):
        return f"{self.user.username}'s Profile"


# ======================================================================
# Model 2: MovieWatchlist
# Relational Concept: One-to-Many Relationship (1:N) & Domain Fields
# Syllabus Topic: Custom Django Models & Database Fields (Unit 9)
# ======================================================================
class MovieWatchlist(models.Model):
    """
    Tracks items saved to a user's watchlist.
    Stores the user reference, unique content tracker ID, and the media type.
    """
    MEDIA_TYPE_CHOICES = [
        ('movie', 'Movie'),
        ('tv', 'TV Show'),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,      # Deletes watchlist entry when user is deleted
        related_name='watchlist'       # Reverse relationship mapping (e.g., user.watchlist.all())
    )
    
    # Store the unique database/TMDB identifier (e.g. 19995 for Avatar)
    media_id = models.IntegerField(
        help_text="The unique tracking ID from the TMDB dataset."
    )
    
    # Media type discriminator string to support dualnavbar pages (Movies and TV tabs)
    media_type = models.CharField(
        max_length=10,
        choices=MEDIA_TYPE_CHOICES,
        default='movie',
        help_text="Discriminator column to distinguish between 'movie' and 'tv' show records."
    )
    
    added_at = models.DateTimeField(
        auto_now_add=True,             # Automatically sets field to current datetime when created
        help_text="Timestamp of when the media was saved to the watchlist."
    )

    class Meta:
        verbose_name = "Movie Watchlist Item"
        verbose_name_plural = "Movie Watchlist Items"
        # Database constraint: Prevent duplicate entries of the same show in a user's watchlist
        unique_together = ('user', 'media_id', 'media_type')

    def __str__(self):
        return f"{self.user.username} saved {self.media_type} ID {self.media_id}"


# ======================================================================
# Model 3: UserReview
# Relational Concept: Foreign Keys (1:N) & Numerical Constraints
# Syllabus Topic: Django Validation, Timestamps & Database Constraints (Unit 9)
# ======================================================================
class UserReview(models.Model):
    """
    Stores written user reviews and numerical ratings for films and TV shows.
    """
    MEDIA_TYPE_CHOICES = [
        ('movie', 'Movie'),
        ('tv', 'TV Show'),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='reviews'
    )
    
    # Target media identifiers
    media_id = models.IntegerField(
        help_text="The unique TMDB database identifier being reviewed."
    )
    media_type = models.CharField(
        max_length=10,
        choices=MEDIA_TYPE_CHOICES,
        default='movie',
        help_text="Distinguishes whether this review belongs to a movie or a TV show."
    )

    # 1-to-5 star integer rating with Min/Max value constraints
    rating = models.IntegerField(
        validators=[
            MinValueValidator(1, message="Rating must be at least 1 star."),
            MaxValueValidator(5, message="Rating cannot exceed 5 stars.")
        ],
        help_text="Integer rating from 1 (lowest) to 5 (highest) stars."
    )
    
    # Written description review text
    review_text = models.TextField(
        help_text="User's detailed comments regarding the movie or TV show."
    )
    
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp of when the review was written."
    )

    class Meta:
        verbose_name = "User Review"
        verbose_name_plural = "User Reviews"
        # Ordering reviews by newest first
        ordering = ['-created_at']

    def __str__(self):
        return f"Review by {self.user.username} on {self.media_type} ID {self.media_id} ({self.rating} stars)"


from django.db import models
from django.contrib.auth.models import User

class MediaReview(models.Model):
    """
    Syllabus Reference: Unit 9.1 Database Relations & Relational Schema Mapping
    Stores structured user reviews for movies/TV shows, acting as an ingestion
    source for future high-dimensional Sentiment Analysis / NLP vector clusters.
    """
    MEDIA_CHOICES = [
        ('movie', 'Movie'),
        ('tv', 'TV Show'),
    ]
    
    # 💎 FIXED: Altered related_name to resolve the reverse accessor namespace clash with UserReview
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='media_reviews')
    media_id = models.IntegerField()  # Corresponds to TMDB API unique IDs
    media_type = models.CharField(max_length=10, choices=MEDIA_CHOICES, default='movie')
    review_text = models.TextField()
    
    # 🧠 Metadata pillars for upcoming analytical aggregations
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Placeholder field for future Phase 5 Machine Learning model execution integration
    sentiment_score = models.FloatField(null=True, blank=True, help_text="Aggregated polarity score [-1.0 to 1.0]")

    class Meta:
        ordering = ['-created_at']  # Show freshest critical insights first
        # Prevent a single user from spamming multiple root records for the same property
        unique_together = ('user', 'media_id', 'media_type')

    def __name__(self):
        return f"{self.user.username} - {self.media_type} {self.media_id} ({self.created_at.strftime('%Y-%m-%d')})"

    def __str__(self):
        return f"{self.user.username} - {self.media_type} {self.media_id} ({self.created_at.strftime('%Y-%m-%d')})"


class Review(models.Model):
    """
    Syllabus Reference: Unit 9.1 Database Relations & Schema Mapping
    Stores user reviews and ratings (1-10) for movies.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='user_custom_reviews')
    movie_id = models.IntegerField(help_text="The unique TMDB movie identifier.")
    movie_title = models.CharField(max_length=255)
    rating = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(10)],
        help_text="Integer rating from 1 (lowest) to 10 (highest)."
    )
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        unique_together = ('user', 'movie_id')

    def __str__(self):
        return f"{self.user.username}'s review of {self.movie_title} ({self.rating}/10)"


class WatchedHistory(models.Model):
    """
    Stores watched movie history to track analytics and total watch duration.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='watched_history')
    movie_id = models.IntegerField()
    movie_title = models.CharField(max_length=255)
    duration = models.IntegerField(help_text="Watch duration in minutes.")
    rating = models.FloatField(null=True, blank=True, help_text="Rating given by user (1 to 10).")
    genres = models.CharField(max_length=255, help_text="Comma-separated genre names.")
    watched_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} watched {self.movie_title}"


class CachedMedia(models.Model):
    """
    Syllabus Reference: Unit 9 (Database Schema Design)
    Acts as a local database cache for movie and TV show detail payloads.
    Decouples user requests from live network API calls for high availability.
    """
    MEDIA_TYPE_CHOICES = [
        ('movie', 'Movie'),
        ('tv', 'TV Show'),
    ]
    media_id = models.IntegerField(help_text="The unique database tracking ID from TMDB.")
    media_type = models.CharField(
        max_length=10,
        choices=MEDIA_TYPE_CHOICES,
        default='movie',
        help_text="Discriminator column ('movie' or 'tv')."
    )
    data = models.JSONField(help_text="Stores the raw TMDb API detail response JSON payload.")
    updated_at = models.DateTimeField(auto_now=True, help_text="Timestamp of when the cache record was synced.")
    is_manual_override = models.BooleanField(
        default=False,
        help_text="Enables overriding the provider data manually with verified streaming networks."
    )
    manual_providers = models.JSONField(
        default=list,
        blank=True,
        null=True,
        help_text="Custom list of providers: [{'name': 'Netflix', 'logo_url': '...', 'web_url': '...'}]"
    )

    class Meta:
        verbose_name = "Cached Media Item"
        verbose_name_plural = "Cached Media Items"
        unique_together = ('media_id', 'media_type')

    def __str__(self):
        return f"Cached {self.media_type.upper()} ID {self.media_id}"


class GenreCache(models.Model):
    """
    Syllabus Reference: Unit 9 (Database Schema Design)
    Stores TMDB discovery results by genre locally in database.
    Bypasses API timeouts during category browsing.
    """
    genre_name = models.CharField(max_length=100, unique=True, help_text="Name of the genre (e.g. Crime, Action).")
    data = models.JSONField(help_text="Stores the JSON list of discovered media items.")
    updated_at = models.DateTimeField(auto_now=True, help_text="Timestamp of when this genre cache was synced.")

    class Meta:
        verbose_name = "Genre Cache"
        verbose_name_plural = "Genre Caches"

    def __str__(self):
        return f"Genre Cache: {self.genre_name}"


class CachedRecommendation(models.Model):
    """
    Stores pre-computed top-15 cosine similarity matches for movies and TV shows.
    Allows low-resource deployment by offloading similarity matrix slicing to standard indexed database queries.
    """
    source_id = models.IntegerField(db_index=True, help_text="ID of the movie/show being viewed.")
    target_id = models.IntegerField(help_text="ID of the recommended movie/show.")
    score = models.FloatField(help_text="Cosine similarity score.")
    media_type = models.CharField(max_length=10, help_text="Discriminator column ('movie' or 'tv').")

    class Meta:
        verbose_name = "Cached Recommendation"
        verbose_name_plural = "Cached Recommendations"
        indexes = [
            models.Index(fields=['source_id', 'media_type']),
        ]

    def __str__(self):
        return f"{self.media_type.upper()} {self.source_id} -> {self.target_id} ({self.score:.2f})"


class RecommendationFeedback(models.Model):
    """A user signal that removes a title from future personalised feeds."""
    MEDIA_TYPE_CHOICES = [
        ('movie', 'Movie'),
        ('tv', 'TV Show'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='recommendation_feedback')
    media_id = models.IntegerField()
    media_type = models.CharField(max_length=10, choices=MEDIA_TYPE_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'media_id', 'media_type')
        indexes = [models.Index(fields=['user', 'media_type'])]

    def __str__(self):
        return f"{self.user.username} is not interested in {self.media_type} {self.media_id}"


# ======================================================================
# Collaborative Watch Groups & Shared Watchlist Models
# ======================================================================
class WatchGroup(models.Model):
    """
    Represents a shared Watch Group (Watch Party) created by a user.
    """
    name = models.CharField(max_length=255, help_text="The name of the watch group.")
    invite_code = models.UUIDField(default=uuid.uuid4, unique=True, editable=False, db_index=True, help_text="Unique invite token.")
    creator = models.ForeignKey(User, on_delete=models.CASCADE, related_name='created_groups', help_text="Owner of the group.")
    members = models.ManyToManyField(User, related_name='watch_groups', help_text="Users belonging to this group.")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Watch Group"
        verbose_name_plural = "Watch Groups"
        ordering = ['-created_at']

    def __str__(self):
        return f"Group: {self.name} (by {self.creator.username})"


class SharedWatchlist(models.Model):
    """
    Represents a movie or TV show added to a group's collaborative watchlist.
    """
    MEDIA_TYPE_CHOICES = [
        ('movie', 'Movie'),
        ('tv', 'TV Show'),
    ]
    group = models.ForeignKey(WatchGroup, on_delete=models.CASCADE, related_name='watchlist_items', help_text="Associated Watch Group.")
    media_id = models.IntegerField(help_text="The TMDB identifier.")
    media_type = models.CharField(max_length=10, choices=MEDIA_TYPE_CHOICES, default='movie')
    title = models.CharField(max_length=255)
    poster_url = models.CharField(max_length=500, blank=True, null=True)
    added_by = models.ForeignKey(User, on_delete=models.CASCADE, help_text="The user who added this title.")
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Shared Watchlist Item"
        verbose_name_plural = "Shared Watchlist Items"
        ordering = ['-added_at']
        unique_together = ('group', 'media_id', 'media_type')

    def __str__(self):
        return f"{self.title} in {self.group.name} (added by {self.added_by.username})"


class GroupMessage(models.Model):
    """
    Represents a chat message sent inside a Watch Group.
    """
    group = models.ForeignKey(WatchGroup, on_delete=models.CASCADE, related_name='messages', help_text="The Watch Group containing this conversation.")
    sender = models.ForeignKey(User, on_delete=models.CASCADE, help_text="The member who sent this message.")
    content = models.TextField(help_text="Body content of the message.")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Group Message"
        verbose_name_plural = "Group Messages"
        ordering = ['created_at']

    def __str__(self):
        return f"Msg from {self.sender.username} in {self.group.name} at {self.created_at.strftime('%Y-%m-%d %H:%M')}"


class Notification(models.Model):
    """
    Represents an in-app notification sent to a user.
    """
    recipient = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications', help_text="The user receiving this notification.")
    sender = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sent_notifications', null=True, blank=True, help_text="The user who triggered the notification.")
    message = models.TextField(help_text="Notification body text description.")
    target_url = models.CharField(max_length=500, help_text="Redirect path URL when notification is clicked.")
    is_read = models.BooleanField(default=False, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Notification"
        verbose_name_plural = "Notifications"
        ordering = ['-created_at']

    def __str__(self):
        return f"Notification to {self.recipient.username}: {self.message[:30]}..."


class ContinueWatching(models.Model):
    """
    Stores a user's recent watch history for movies and TV episodes.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='continue_watching_history')
    media_id = models.CharField(max_length=50)
    media_type = models.CharField(max_length=10, choices=[('movie', 'Movie'), ('tv', 'TV Show')])
    title = models.CharField(max_length=255)
    poster_url = models.CharField(max_length=500, blank=True, null=True)
    season = models.IntegerField(null=True, blank=True)
    episode = models.IntegerField(null=True, blank=True)
    episode_title = models.CharField(max_length=255, blank=True, null=True)
    last_watched = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Continue Watching"
        verbose_name_plural = "Continue Watching Logs"
        ordering = ['-last_watched']
        unique_together = ('user', 'media_id', 'media_type')

    def __str__(self):
        if self.media_type == 'tv':
            return f"{self.user.username} is watching {self.title} S{self.season} E{self.episode}"
        return f"{self.user.username} is watching {self.title}"

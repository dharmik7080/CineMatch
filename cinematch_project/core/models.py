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
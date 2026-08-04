"""
CineMatch Project - AppConfig
Syllabus Reference: Units 4 & 5 (Model Loading & Caching Optimization)

The ready() hook fires ONCE when Django finishes setting up the full
application registry (after all models and middleware are initialized).
Loading the vector similarity pickles here moves the blocking disk I/O
completely out of the HTTP request/response cycle, eliminating the
multi-second navbar freeze that occurred when load_ml_models() was
triggered lazily on the first incoming request.
"""

from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        """
        Triggered once at server boot after the full app registry is loaded.
        Pre-loads all ML vector pickles into process memory so every subsequent
        HTTP request reads from RAM (O(1)) rather than disk (O(n seconds)).
        """
        # Deferred import: avoids circular imports during Django's boot sequence.
        # Views module is only safe to import after apps are fully registered.
        try:
            from .views import load_ml_models
            load_ml_models()
        except Exception as e:
            # Non-fatal: server still starts if pickle files are missing;
            # recommendation views will return empty lists gracefully.
            print(f"[APPCONFIG] Warning: ML model pre-load failed at boot: {e}")

from django.conf import settings
from core.models import Notification

def notifications_unread_count(request):
    """
    Injected globally to all rendering context scopes.
    Returns the integer count of unread notifications for request.user.
    """
    if request.user and request.user.is_authenticated:
        count = Notification.objects.filter(recipient=request.user, is_read=False).count()
        return {'unread_notifications_count': count}
    return {'unread_notifications_count': 0}

def analytics_context(request):
    """
    Exposes Google Analytics 4 Measurement ID globally to base templates.
    """
    return {
        'GA_MEASUREMENT_ID': getattr(settings, 'GA_MEASUREMENT_ID', '')
    }

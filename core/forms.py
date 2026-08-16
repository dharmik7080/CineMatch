from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django_recaptcha.fields import ReCaptchaField
from django_recaptcha.widgets import ReCaptchaV2Invisible

class CineMatchRegistrationForm(UserCreationForm):
    captcha = ReCaptchaField(widget=ReCaptchaV2Invisible)

class CineMatchLoginForm(AuthenticationForm):
    captcha = ReCaptchaField(widget=ReCaptchaV2Invisible)

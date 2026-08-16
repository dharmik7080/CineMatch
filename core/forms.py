from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django_recaptcha.fields import ReCaptchaField
from django_recaptcha.widgets import ReCaptchaV3

class CineMatchRegistrationForm(UserCreationForm):
    captcha = ReCaptchaField(widget=ReCaptchaV3)

class CineMatchLoginForm(AuthenticationForm):
    captcha = ReCaptchaField(widget=ReCaptchaV3)

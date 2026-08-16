from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django_recaptcha.fields import ReCaptchaField

class CineMatchRegistrationForm(UserCreationForm):
    captcha = ReCaptchaField()

class CineMatchLoginForm(AuthenticationForm):
    captcha = ReCaptchaField()

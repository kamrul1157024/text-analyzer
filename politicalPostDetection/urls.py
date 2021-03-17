from django.urls import path
from politicalPostDetection import views

urlpatterns = [
    path("api/isPoliticalPost/", views.isPoliticalPost, name="political-post-identifier"),
    path("api/detectLanguage/", views.detectLanguage, name="language-detector")
]

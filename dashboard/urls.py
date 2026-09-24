from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('update-email/', views.update_email_view, name='update_email'),
    path('api/delete-account/', views.delete_account_view, name='delete_account'),
    path('api/update-username/', views.update_username_view, name='update_username'),
    path('api/update-profile/', views.update_profile_view, name='update_profile'),
    path('api/chat/', views.api_chat, name='api_chat'),
    path('api/chats/', views.api_chats, name='api_chats'),
    path('api/image-generate/', views.api_image_generate, name='api_image_generate'),
]

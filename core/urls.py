from django.contrib import admin
from django.urls import path
from dashboard.views import (
    index, 
    login_view, 
    register_view, 
    logout_view, 
    delete_account_view, 
    api_chat,
    update_username_view
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),
    path('login/', login_view, name='login'),
    path('register/', register_view, name='register'),
    path('logout/', logout_view, name='logout'),
    path('api/delete-account/', delete_account_view, name='delete_account'),
    path('api/chat/', api_chat, name='api_chat'),
    path('api/update-username/', update_username_view, name='update_username'),
]
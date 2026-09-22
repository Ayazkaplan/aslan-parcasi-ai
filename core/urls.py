from django.contrib import admin
from django.urls import path
from dashboard.views import (
    index, 
    login_view, 
    register_view, 
    logout_view, 
    update_email_view, 
    delete_account_view, 
    api_chat
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),
    path('login/', login_view, name='login'),
    path('register/', register_view, name='register'),
    path('logout/', logout_view, name='logout'),
    path('update-email/', update_email_view, name='update_email'),
    path('api/delete-account/', delete_account_view, name='delete_account'),
    path('api/chat/', api_chat, name='api_chat'),
]
from django.core.management.base import BaseCommand
from django.utils import timezone
from dashboard.models import ChatHistory, UserProfile
from django.contrib.auth.models import User
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Veritabanı tarihlerini güncel tarihe günceller'

    def handle(self, *args, **options):
        current_time = timezone.now()
        self.stdout.write(f"Veritabanı tarihleri güncelleniyor... Şu anki zaman: {current_time}")
        
        # ChatHistory güncelle
        chat_count = ChatHistory.objects.count()
        if chat_count > 0:
            ChatHistory.objects.update(updated_at=current_time)
            self.stdout.write(self.style.SUCCESS(f"✓ {chat_count} ChatHistory kaydı güncellendi"))
        
        # UserProfile güncelle
        profile_count = UserProfile.objects.count()
        if profile_count > 0:
            UserProfile.objects.update(updated_at=current_time)
            self.stdout.write(self.style.SUCCESS(f"✓ {profile_count} UserProfile kaydı güncellendi"))
        
        # User.last_login güncelleme (opsiyonel)
        user_count = User.objects.count()
        if user_count > 0:
            User.objects.filter(last_login__isnull=False).update(last_login=current_time)
            self.stdout.write(self.style.SUCCESS(f"✓ {user_count} User kaydı güncellendi"))
        
        self.stdout.write(self.style.SUCCESS("Veritabanı tarihleri başarıyla güncellendi!"))

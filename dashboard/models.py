from django.db import models
from django.contrib.auth.models import User

class Proje(models.Model):
    baslik = models.CharField(max_length=200, verbose_name="Proje Başlığı")
    aciklama = models.TextField(verbose_name="Açıklama")
    olusturulma_tarihi = models.DateTimeField(auto_now_add=True, verbose_name="Oluşturulma Tarihi")
    durum = models.BooleanField(default=True, verbose_name="Aktif mi?")

    def __str__(self):
        return self.baslik

    class Meta:
        verbose_name = "Proje"
        verbose_name_plural = "Projeler"


class UserProfile(models.Model):
    """Per-user settings that must survive new devices and deployments."""

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="profile",
    )
    avatar = models.TextField(blank=True, default="")
    theme = models.CharField(max_length=40, default="theme-cyber")
    pattern = models.CharField(max_length=40, default="pattern-grid")
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username} profili"
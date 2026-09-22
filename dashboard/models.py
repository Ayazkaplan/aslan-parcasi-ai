from django.db import models

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
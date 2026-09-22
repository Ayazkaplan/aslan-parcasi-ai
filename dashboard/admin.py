from django.contrib import admin
from .models import Proje

# Admin panelinin mavi başlıklarını özelleştiriyoruz
admin.site.site_header = "Aslan Parçası Yönetim Paneli"
admin.site.site_title = "Aslan Parçası Admin"
admin.site.index_title = "Yönetim Paneline Hoş Geldiniz"

# Oluşturduğumuz Proje tablosunu admin paneline kaydediyoruz
admin.site.register(Proje)
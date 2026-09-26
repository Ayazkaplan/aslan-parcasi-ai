from django.conf import settings
from django.contrib.auth import login
from django.contrib.auth.models import User
from django.core import signing

REMEMBER_COOKIE = "aslan_remember"
REMEMBER_MAX_AGE = 60 * 60 * 24 * 90  # 90 gün
REMEMBER_SALT = "aslan-parcasi-remember-me"


def build_remember_token(user):
    return signing.dumps(user.pk, salt=REMEMBER_SALT)


def set_remember_cookie(response, user):
    response.set_cookie(
        REMEMBER_COOKIE,
        build_remember_token(user),
        max_age=REMEMBER_MAX_AGE,
        httponly=True,
        secure=not settings.DEBUG,
        samesite="Lax",
    )
    return response


def clear_remember_cookie(response):
    response.delete_cookie(REMEMBER_COOKIE)
    return response


class AutoLoginMiddleware:
    """Session kaybolduğunda (ör. deploy sonrası) imzalı çerezden kullanıcıyı otomatik giriş yapar."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not request.user.is_authenticated:
            token = request.COOKIES.get(REMEMBER_COOKIE)
            if token:
                user = None
                try:
                    user_id = signing.loads(
                        token, max_age=REMEMBER_MAX_AGE, salt=REMEMBER_SALT
                    )
                    user = User.objects.filter(pk=user_id, is_active=True).first()
                except signing.BadSignature:
                    pass
                if user is not None:
                    login(request, user)
                    request.session.set_expiry(settings.SESSION_COOKIE_AGE)
        return self.get_response(request)

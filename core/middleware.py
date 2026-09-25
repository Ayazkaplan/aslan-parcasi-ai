from django.conf import settings


class TrustRequestOriginMiddleware:
    """Accept CSRF from the current host so preview/deploy URLs keep sessions."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        host = request.get_host().split(":")[0]
        if host:
            scheme = "https" if request.is_secure() else "http"
            origins = [
                f"{scheme}://{request.get_host()}",
                f"https://{host}",
                f"http://{host}",
            ]
            trusted = list(getattr(settings, "CSRF_TRUSTED_ORIGINS", []))
            for origin in origins:
                if origin not in trusted:
                    trusted.append(origin)
            settings.CSRF_TRUSTED_ORIGINS = trusted
        return self.get_response(request)

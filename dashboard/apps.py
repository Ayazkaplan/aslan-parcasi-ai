import os
import sys
import threading
import time

from django.apps import AppConfig


CLOCK_REFRESH_SECONDS = 3600
_clock_thread_started = False


def _refresh_clock_once():
    from django.utils import timezone
    from .models import AppClock

    current_date = timezone.localdate()
    clock, _ = AppClock.objects.get_or_create(
        singleton=True,
        defaults={"current_date": current_date},
    )
    if clock.current_date != current_date:
        clock.current_date = current_date
    clock.save(update_fields=["current_date", "updated_at"])
    return clock


def _hourly_clock_loop():
    while True:
        time.sleep(CLOCK_REFRESH_SECONDS)
        try:
            _refresh_clock_once()
        except Exception:
            # The hourly refresh must never take the web process down.
            pass


class DashboardConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'dashboard'

    def ready(self):
        global _clock_thread_started
        if _clock_thread_started:
            return
        if threading.current_thread() is not threading.main_thread():
            return
        # Under `runserver` only the reloader child should start the thread.
        if "runserver" in sys.argv and os.environ.get("RUN_MAIN") != "1":
            return
        _clock_thread_started = True
        thread = threading.Thread(target=_hourly_clock_loop, name="aslan-clock-refresh", daemon=True)
        thread.start()

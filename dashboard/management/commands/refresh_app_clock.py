from django.core.management.base import BaseCommand

from dashboard.views import sync_app_clock


class Command(BaseCommand):
    help = "Persist the current Europe/Istanbul calendar date."

    def handle(self, *args, **options):
        clock = sync_app_clock()
        self.stdout.write(
            self.style.SUCCESS(
                f"AppClock güncellendi: {clock.current_date.isoformat()}"
            )
        )
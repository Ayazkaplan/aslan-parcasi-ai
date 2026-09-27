from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("dashboard", "0003_chathistory"),
    ]

    operations = [
        migrations.CreateModel(
            name="AppClock",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("singleton", models.BooleanField(default=True, unique=True)),
                ("current_date", models.DateField()),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
        ),
    ]
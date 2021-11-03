# Generated by Django 3.2.8 on 2021-11-03 01:42

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='EfficientData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pub_date', models.DateTimeField(default=django.utils.timezone.now, verbose_name='日付')),
                ('photo_image', models.ImageField(upload_to='images/%Y%m%d/', verbose_name='画像')),
                ('pred_result', models.CharField(max_length=200, verbose_name='推論結果')),
            ],
        ),
    ]

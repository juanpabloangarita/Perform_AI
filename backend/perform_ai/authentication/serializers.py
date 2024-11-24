from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

from perform_ai.authentication.models import User


class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, validators=[validate_password])

    class Meta:
        model = User
        fields = "__all__"

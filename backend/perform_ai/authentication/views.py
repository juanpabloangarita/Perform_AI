from django.http import JsonResponse
from django.contrib.auth.views import LoginView as DefaultLoginView

class LoginView(DefaultLoginView):
    def form_valid(self, form):
        response = super().form_valid(form)
        return JsonResponse({'message': 'You have successfully logged in.'})

    def form_invalid(self, form):
        response = super().form_invalid(form)
        return JsonResponse({'message': 'Invalid credentials.'}, status=401)


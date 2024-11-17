from django.shortcuts import render

def index(request):
    return render(request, 'index.html')

def prompt_interface(request):
    return render(request, 'ai_interface/prompt_interface.html')
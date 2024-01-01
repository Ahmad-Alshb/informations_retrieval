# -*- coding: utf-8 -*-
from django.db import models
class Documentdata(models.Model):
 
    file_name = models.CharField(max_length=100)
    content = models.TextField()
    language = models.CharField(max_length=50)
    weighting_algorithm = models.CharField(max_length=50)
  


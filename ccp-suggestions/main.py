from os import environ
from src.commands.image_prediction import PredictionModel
from src.commands.generate_recommendation import GenerateRecommendations
from src.static.constants import json_key

environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key
PredictionModel().execute()
r = GenerateRecommendations().execute()
print(r)
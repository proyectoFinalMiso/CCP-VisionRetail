from dotenv import load_dotenv
from os import environ
from src.commands.image_prediction import PredictionModel
from src.commands.generate_recommendation import GenerateRecommendations
from src.commands.common.pubsub import pull_single_message
from src.static.constants import json_key

load_dotenv('.env')

environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key
# PredictionModel().execute()
# r = GenerateRecommendations().execute()
r = pull_single_message('video-processing-sub')
print(r)
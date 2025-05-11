from flask import Blueprint, jsonify, request
from src.commands.image_prediction import PredictionModel
from src.commands.generate_recommendation import GenerateRecommendations
from src.commands.send_email import SendEmail

blueprint = Blueprint('recommendations', __name__)

@blueprint.get('/ping')
def health_check():
    return jsonify({'message': 'El servicio está activo'}), 200

@blueprint.post('/predict')
def predict_model():
    body = request.get_json()
    print(body)
    r = PredictionModel(body).execute()
    return jsonify(r['response']), r['status_code']

@blueprint.post('/recommendations')
def create_recommendations():
    body = request.get_json()
    print(body)
    r = GenerateRecommendations(body).execute()
    return jsonify(r['response']), r['status_code']

@blueprint.post('/mail_recommendations')
def send_recommendations():
    body = request.get_json()
    print(body)
    r = SendEmail(body).execute()
    return jsonify(r['response']), r['status_code']
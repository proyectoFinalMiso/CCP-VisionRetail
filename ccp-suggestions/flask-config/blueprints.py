from flask import Blueprint, jsonify, request
from src.commands.image_prediction import PredictionModel


blueprint = Blueprint('recommendations', __name__)

@blueprint.get('/ping')
def health_check():
    return jsonify({'message': 'El servicio está activo'}), 200

@blueprint.post('generate')
def generate_suggestion():
    body = request.get_json()
    r = PredictionModel(body)
    return jsonify(r['response']), r['status_code']

@blueprint.post('mail')
def email_suggestion():
    body = request.get_json()
    pass
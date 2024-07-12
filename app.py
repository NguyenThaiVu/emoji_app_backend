import yaml
import joblib
from flask import Flask, request, jsonify
import os
from comet_ml import API
from utils.utils_model import *

app = Flask(__name__)


# The configuration
with open("config.yaml", 'r') as file:  
    config = yaml.safe_load(file)


def download_model_from_comet(config, registry_name, output_folder="models", version='1.0.0'):
    """
    This function download the saved model from Comet

    * Parameter:
    registry_name (str) -- the name of saved model. 'pretrain_glove_model' or 'emojify_xgb_classification'
    """

    api = API(api_key=config['programming']['comet_api_key'])

    api.download_registry_model(
        workspace=config['programming']['comet_workspace'],
        registry_name=registry_name,
        version=version,
        output_path=output_folder
    )


# Load model
path_file_pretrain_glove = config['model']['glove']['path_file']
if os.path.exists(path_file_pretrain_glove):
    glove_embed = load_glove_embeddings(path_file_pretrain_glove)
else:
    download_model_from_comet(config, registry_name='pretrain_glove_model', version=config['model']['glove']['current_version'])
    glove_embed = load_glove_embeddings(path_file_pretrain_glove)

path_file_trained_xgb_model = config['model']['xgb_model']['path_file']
if os.path.exists(path_file_trained_xgb_model):
    xgb_model = joblib.load(path_file_trained_xgb_model)
else:
    download_model_from_comet(config, registry_name='emojify_xgb_classification', version=config['model']['xgb_model']['current_version'])
    xgb_model = joblib.load(path_file_trained_xgb_model)

path_file_label_encoder = config['model']['label_encoder']['path_file']
if os.path.exists(path_file_label_encoder):
    label_encoder = joblib.load(path_file_label_encoder)
else:
    download_model_from_comet(config, registry_name='emojify_label_encoder', version=config['model']['xgb_model']['current_version'])
    label_encoder = joblib.load(path_file_label_encoder)


top_k = config['prediction']['top_k']



@app.route('/predict', methods=['POST'])
def predict():
    """
    This function handle the post request from frontend
    """

    # Get RAW input data
    data = request.get_json()
    input_text = data['input_text']

    # Run inference 
    input_text = str(input_text)
    list_predcited_emotion = get_top_k_prediction(xgb_model, input_text, top_k, glove_embed, label_encoder)

    output_prediction = {'list_predcited_emotion': list_predcited_emotion}
    return jsonify(output_prediction)


if __name__ == '__main__':
    # app.run(port=5000, debug=True)
    app.run(debug=True)
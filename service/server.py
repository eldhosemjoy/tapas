#!/usr/bin/env python
import json
from pathlib import Path
from falcon_cors import CORS
import falcon
from falcon_multipart.middleware import MultipartMiddleware
from .tapasPredictor.tapasPredictorService import TapasPredictorService, Root, TapasRoot

def load_config(config_path):
    config = {}
    with open(config_path) as f:
        config = json.load(f)
    return config
    

try:
    unicode
except NameError:
    unicode = str

cors = CORS(allow_all_origins=True, allow_all_methods=True, allow_all_headers=True)

#Routes
config = load_config("config.json")
PORT =  config['port']
APP  = falcon.API(middleware=[cors.middleware,MultipartMiddleware()])
APP.add_route('/', Root())
APP.add_route('/tapas_predictor', TapasPredictorService())
APP.add_route('/tapas', TapasRoot())


print("------------------------------------------------------------------------------------------")
print("--------------------TAPAS Preditor started at port "+str(PORT)+"-----------------------------------")
print("------------------------------------------------------------------------------------------")



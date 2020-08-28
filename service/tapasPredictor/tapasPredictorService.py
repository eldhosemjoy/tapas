from __future__ import unicode_literals
from __future__ import print_function
import json
import falcon
import re
import uuid
import configparser
import traceback
import os
import traceback

from tapas_predictor import TapasPredictor		
tapaspredictor 	= TapasPredictor()

class Root(object):
	def on_get(self, req, resp):
		try:
			resp.content_type = 'text/html'
			resp.status = falcon.HTTP_200
			html = "<html><body><h1>TAPAS Predictor!</h1></body></html>"
			resp.body = html
		except Exception as e:
			raise falcon.HTTPBadRequest(
				'Root could not be rendered',
				'{}'.format(e))
			resp.status = falcon.HTTP_500

class TapasPredictorService(object):
	def on_post(self, req, resp):
		try:
			request_json 	= json.load(req.bounded_stream,encoding='utf-8')
			question 		= request_json['question']
			table			= request_json['table']
			queries = [question]
			prediction 		= tapaspredictor.predict(table,queries)
			answer			= {"question":question, "prediction":prediction}
			response_json = json.dumps(answer)

			print(response_json)

			resp.body  = response_json
			resp.content_type = 'application/json'
			resp.status = falcon.HTTP_200
		except Exception as e:
			traceback.print_exc()
			raise falcon.HTTPBadRequest(
				'Tapas Predicted failed!',
				'{}'.format(e))
			resp.status = falcon.HTTP_500

class TapasRoot(object):
	def on_get(self, req, resp):
		try:
			table = [
						["Pos", "No", "Driver","Team","Laps","Time/Retired","Grid","Points"],
						["1","32","Patrick Carpentier","Team Player's","87","1:48:11.023","1","22"],
						["2","1","Bruno Junqueira","Newman/Haas Racing","87","+0.8 secs","2","17"],
						["3","3","Paul Tracy","Team Player's","87","+28.6 secs","3","14"],
						["4","9","Michel Jourdain, Jr.","Team Rahal","87","+40.8 secs","13","12"],
						["5","34","Mario Haberfeld","Mi-Jack Conquest Racing","87","+42.1 secs","6","10"],
						["6","20","Oriol Servia","Patrick Racing","87","+1:00.2","10","8"],
						["7","51","Adrian Fernandez","Fernandez Racing","87","+1:01.4","5","6"],
						["8","12","Jimmy Vasser","American Spirit Team Johansson","87","+1:01.8","8","5"],
						["9","7","Tiago Monteiro","Fittipaldi-Dingman Racing","86","+ 1 Lap","15","4"],
						["10","55","Mario Dominguez","Herdez Competition","86","+ 1 Lap","11","3"],
						["11","27","Bryan Herta","PK Racing","86","+ 1 Lap","12","2"],
						["12","31","Ryan Hunter-Reay","American Spirit Team Johansson","86","+ 1 Lap","17","1"],
						["13","19","Joel Camathias","Dale Coyne Racing","85","+ 2 Laps","18","0"],
						["14","33","Alex Tagliani","Rocketsports Racing","85","+ 2 Laps","14","0"],
						["15","4","Roberto Moreno","Herdez Competition","85","+ 2 Laps","9","0"],
						["16","11","Geoff Boss","Dale Coyne Racing","83","Mechanical","19","0"],
						["17","2","Sebastien Bourdais","Newman/Haas Racing","77","Mechanical","4","0"],
						["18","15","Darren Manning","Walker Racing","12","Mechanical","7","0"],
						["19","5","Rodolfo Lavin","Walker Racing","10","Mechanical","16","0"]
					]
			queries = ["what were the drivers names?"]
			prediction 		= tapaspredictor.predict(table,queries)
			answer			= {"question":queries[0], "prediction":prediction}
			response_json = json.dumps(answer)
			resp.content_type = 'text/html'
			resp.status = falcon.HTTP_200
			html = "<html><body><h1>TAPAS Predictor!</h1><br>"+response_json+"</body></html>"
			resp.body = html
		except Exception as e:
			traceback.print_exc()
			raise falcon.HTTPBadRequest(
				'Query could not be answered!',
				'{}'.format(e))
			resp.status = falcon.HTTP_500
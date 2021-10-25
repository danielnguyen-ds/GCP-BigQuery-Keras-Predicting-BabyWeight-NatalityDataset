import os

from flask import Flask
from flask import render_template
from flask import request
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials


credentials = GoogleCredentials.get_application_default()
api = discovery.build("ml", "v1", credentials=credentials)

app = Flask(__name__)


def get_prediction(features):
	project = os.getenv("PROJECT", "predict-babyweight-10142021")
	model_name = os.getenv("MODEL_NAME", "babyweight") 
	version_name = os.getenv("VERSION_NAME", "v1")

	input_data = {"instances": [features]}
	
	# Write a formatted string to make a prediction against a CAIP deployed model.
	parent = f"projects/{project}/models/{model_name}/versions/{version_name}"
	prediction = api.projects().predict(body=input_data, name=parent).execute()

	return prediction["predictions"][0]["weight"][0]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
	def gender2str(val):      
		genders = {"unknown": "Unknown", "male": "True", "female": "False"}
		return genders[val]

	def plurality2str(val):       
		pluralities = {"1": "Single(1)", "2": "Twins(2)", "3": "Triplets(3)","4": "Quadruplets(4)", "5": "Quintuplets(5)"}
		if features["is_male"] == "Unknown" and int(val) > 1:
			return "Multiple(2+)"
		return pluralities[val]

	def cigarette_use2str(val):      
		cigarette_uses = {"unknown": "Unknown", "yes": "True", "no": "False"}
		return cigarette_uses[val]	

	def alcohol_use2str(val):      
		alcohol_uses = {"unknown": "Unknown", "yes": "True", "no": "False"}
		return alcohol_uses[val]	
	
	data = request.form.to_dict()
	mandatory_items = ["babyGender",
                       "motherAge",
					   "plurality",
                       "gestationWeeks",
					   "cigaretteUse",
					   "alcoholUse"]
	for item in mandatory_items:
		if item not in data.keys():
			return "Set all items."

	features = {}
	features["is_male"] = gender2str(data["babyGender"])
	features["mother_age"] = float(data["motherAge"])
	features["plurality"] = plurality2str(data["plurality"])
	features["gestation_weeks"] = float(data["gestationWeeks"])
	features["cigarette_use"] = cigarette_use2str(data["cigaretteUse"])
	features["alcohol_use"] = alcohol_use2str(data["alcoholUse"])

	prediction = get_prediction(features)

	return f"{prediction:.2f} lbs."


if __name__ == '__main__':
	app.run()  # This is used when running locally

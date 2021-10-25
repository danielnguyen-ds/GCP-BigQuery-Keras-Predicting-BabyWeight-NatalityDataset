# End-to-End-ML-Project: GCP-BigQuery-Keras-Predicting-BabyWeight-NatalityDataset
- In this end-to-end Machine Learning project, we use the  [Natality dataset](https://console.cloud.google.com/bigquery?project=bigquery-public-data&p=publicdata&d=samples&t=natality&page=table), which has details on US births from 1969 to 2008 and is a publicly available dataset in [BigQuery](https://cloud.google.com/bigquery/), to build and deploy a [Deep-Wide-Cross Keras model](https://keras.io/examples/structured_data/wide_deep_cross_networks/) *at scale* on [Google Cloud AI Platform](https://cloud.google.com/ai-platform/docs/technical-overview) to predict the weight for a baby before he/she is born. At the end, a [Flask](https://www.fullstackpython.com/flask.html) web form application is then created to show how the model can interact with a web application to provide online predictions.
  
<p align="center">
<img src="assets/Screen_Recording_Prediction.gif" width="500"/>
</p>

- The [Natality dataset](https://console.cloud.google.com/bigquery?project=bigquery-public-data&p=publicdata&d=samples&t=natality&page=table) is a relatively large dataset that has almost 138 million rows and 31 columns in which `weight_pounds` is the target - a continuous value - we train a model to predict, based on values in the other relevant feature columns. So, this is typically a **supervised regression problem**.
- To complete this project, we separate the work into *six* Jupiter notebooks. Each notebook performs a key step in an ***End-to-end Machine Learning Project with a large dataset***:
  - `1_explore_full_dataset.ipynb`: explores and visualizes the Natality dataset using BigQuery calls.
  - `2_prototype_model.ipynb`: uses BigQuery and Pandas to create a small subsample dataset (~20,000 instances) and then uses this sub-dataset to develop a Deep-Wide-Cross Keras model locally.
  - `3_create_ML_datasets.ipynb`: performs data augmentation and preprocessing on the entire Natality dataset, then creates and exports the training/evaluation/testing datasets as CSV files by using two different tools: [Cloud Dataflow](https://cloud.google.com/dataflow) and [BigQuery](https://cloud.google.com/bigquery/).
  - `4_train_model_using_Cloud_AI_Platform`: packages the TensorFlow code up as a Python package and submits the package to [Google Cloud AI Platform](https://cloud.google.com/ai-platform/docs/technical-overview) to train the model at scale with the hyperparameter tuning.
  - `5_serve_online_predictions_with_Cloud_AI_Platform.ipynb`: creates a model version resource in AI Platform that will use our model to serve predictions.
  - `6_deploy_model_with_Flask.ipynb`: makes a [Flask](https://www.fullstackpython.com/flask.html) web form application to show how our model can interact with a web application for the deployment.

***Hope you enjoy it!!!***

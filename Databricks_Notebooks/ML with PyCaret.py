# Databricks notebook source


# COMMAND ----------

from pycaret.classification import *

# COMMAND ----------

from pycaret.datasets import get_data
dataset = get_data('credit')

# COMMAND ----------

data = dataset.sample(frac=0.95, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

# COMMAND ----------

exp_clf101 = setup(data = data, target = 'default', session_id=123)

# COMMAND ----------

rf = create_model('rf')

# COMMAND ----------

tuned_rf = tune_model(rf)

# COMMAND ----------

predict_model(tuned_rf);

# COMMAND ----------

#
final_rf = finalize_model(tuned_rf)

# COMMAND ----------

#Final Random Forest model parameters for deployment
print(final_rf)

# COMMAND ----------

predict_model(final_rf);

# COMMAND ----------

# MAGIC %md
# MAGIC #Deploy Model on Microsoft Azure

# COMMAND ----------

#azure-storage-blob


# COMMAND ----------

## Enter connection string when running in Azure
connect_str = '' #@param {type:"string"}
print(connect_str)

# COMMAND ----------

#! export AZURE_STORAGE_CONNECTION_STRING=connect_str

# COMMAND ----------

os.environ['AZURE_STORAGE_CONNECTION_STRING']= connect_str

# COMMAND ----------


! echo $AZURE_STORAGE_CONNECTION_STRING

# COMMAND ----------

os.getenv('AZURE_STORAGE_CONNECTION_STRING')

# COMMAND ----------

authentication = {'container': 'pycaret-cls-101'}
model_name = 'rf-clf-101'
deploy_model(final_rf, model_name, authentication, platform = 'azure')

# COMMAND ----------


authentication = {'container': 'pycaret-cls-101'}
model_name = 'rf-clf-101'
model_azure = load_model(model_name, 
               platform = 'azure', 
               authentication = authentication,
               verbose=True)

# COMMAND ----------

authentication = {'container': 'pycaret-cls-101'}
model_name = 'rf-clf-101'
unseen_predictions = predict_model(model_name, data=data_unseen, platform='azure', authentication=authentication, verbose=True)

# COMMAND ----------

unseen_predictions

# COMMAND ----------

# MAGIC %md
# MAGIC # PyCaret Eksempel
# MAGIC Reference: https://github.com/pycaret/pycaret/blob/master/examples/PyCaret%202%20Classification.ipynb

# COMMAND ----------

# check version
from pycaret.utils import version
version()

# COMMAND ----------

from pycaret.datasets import get_data
index = get_data('index')

# COMMAND ----------

data = get_data('juice')

# COMMAND ----------

from pycaret.classification import *
clf1 = setup(data = data, target = 'Purchase', session_id=123, log_experiment=True, experiment_name='/Users/kurho@dmpmst.onmicrosoft.com/juice1', verbose=True, silent=True)

# COMMAND ----------

clf1

# COMMAND ----------

best_model = compare_models()

# COMMAND ----------

rf = create_model('rf', fold = 5)

# COMMAND ----------

rf

# COMMAND ----------

tuned_rf = tune_model(rf)

# COMMAND ----------

plot_model(rf)

# COMMAND ----------

plot_model(rf, plot = 'confusion_matrix')

# COMMAND ----------

plot_model(rf, plot = 'boundary')

# COMMAND ----------

plot_model(rf, plot = 'feature')

# COMMAND ----------

plot_model(rf, plot = 'pr')

# COMMAND ----------

plot_model(rf, plot = 'class_report')

# COMMAND ----------

evaluate_model(rf)

# COMMAND ----------

catboost = create_model('catboost', cross_validation=False)

# COMMAND ----------

interpret_model(catboost) # kræver shap

# COMMAND ----------

interpret_model(catboost, plot = 'correlation')

# COMMAND ----------

import shap
shap.initjs()

# COMMAND ----------

interpret_model(catboost, plot = 'reason', observation = 12)

# COMMAND ----------

# loading dataset
from pycaret.datasets import get_data
data = get_data('diabetes')

# initializing setup
from pycaret.classification import *
clf1 = setup(data, target = 'Class variable', log_experiment = True, experiment_name = 'diabetes1')

# compare all baseline models and select top 5
top5 = compare_models(n_select = 5) 

# tune top 5 base models
tuned_top5 = [tune_model(i) for i in top5]

# ensemble top 5 tuned models
bagged_top5 = [ensemble_model(i) for i in tuned_top5]

# blend top 5 base models 
blender = blend_models(estimator_list = top5) 

# run mlflow server (notebook)
!mlflow ui

### just 'mlflow ui' when running through command line.

# COMMAND ----------


best = automl(optimize = 'Recall')
best

# COMMAND ----------


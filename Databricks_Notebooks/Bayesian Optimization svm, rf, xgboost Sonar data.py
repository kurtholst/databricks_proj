# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Bayesian Optimization & Supervised Machine Learning

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview
# MAGIC 1. Load sonar all data
# MAGIC 2. Prepared Data
# MAGIC 3. Split data into training and testing datasets
# MAGIC 4. Encode data for Xgboost (matrix)
# MAGIC 5. 

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/sonar_all_data.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

temp_table_name = "sonar_all_data_csv"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC 
# MAGIC select * from `sonar_all_data_csv`

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "sonar_all_data_csv"

df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

# Import Libraries.

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC

from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours



# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Synthetic binary classification dataset.

# COMMAND ----------

# Generate Synthetic binary classification dataset.
# Only needed if Sonar data is not loaded.
def get_data():
    data, targets = make_classification(
        n_samples=3000,
        n_features=25,
        n_informative=12,
        n_redundant=7,
        random_state=134985745,
    )
    return data, targets

data, targets = get_data()

data.shape
targets[0:10]

# COMMAND ----------

##########################
# Load dataset fra min github
from pandas import read_csv
url = 'https://raw.githubusercontent.com/kurtholst/databricks_proj/master/sonar.all-data.csv'
dataset = read_csv(url, header=None)
dataset

# COMMAND ----------

# Split-out validation dataset
array = dataset.values
data = array[:,0:60].astype(float)
targets = array[:,60]


# COMMAND ----------

###################### Support Vector Machine Classification ###########
def svc_cv(C, gamma, data, targets):
    """SVC cross validation.
    This function will instantiate a SVC classifier with parameters C and
    gamma. Combined with data and targets this will in turn be used to perform
    cross validation. The result of cross validation is returned.
    Our goal is to find combinations of C and gamma that maximizes the roc_auc
    metric.
    """
    estimator = SVC(C=C, gamma=gamma, random_state=2)
    cval = cross_val_score(estimator, data, targets, scoring='roc_auc', cv=4)
    return cval.mean()

def optimize_svc(data, targets):
    """Apply Bayesian Optimization to SVC parameters."""
    def svc_crossval(expC, expGamma):
        """Wrapper of SVC cross validation.
        Notice how we transform between regular and log scale. While this
        is not technically necessary, it greatly improves the performance
        of the optimizer.
        """
        C = 10 ** expC
        gamma = 10 ** expGamma
        return svc_cv(C=C, gamma=gamma, data=data, targets=targets)

    optimizer = BayesianOptimization(
        f=svc_crossval,
        pbounds={"expC": (-3, 2), "expGamma": (-4, -1)},
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=10)

    print("Final result:", optimizer.max)


      
    

# COMMAND ----------

###################### Random Forest Classification ###########
def rfc_cv(n_estimators, min_samples_split, max_features, data, targets):
    """Random Forest cross validation.
    This function will instantiate a random forest classifier with parameters
    n_estimators, min_samples_split, and max_features. Combined with data and
    targets this will in turn be used to perform cross validation. The result
    of cross validation is returned.
    Our goal is to find combinations of n_estimators, min_samples_split, and
    max_features that minimzes the log loss.
    """
    estimator = RFC(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=2
    )
    cval = cross_val_score(estimator, data, targets,
                           scoring='neg_log_loss', cv=4)
    return cval.mean()

def optimize_rfc(data, targets):
    """Apply Bayesian Optimization to Random Forest parameters."""
    def rfc_crossval(n_estimators, min_samples_split, max_features):
        """Wrapper of RandomForest cross validation.
        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """
        return rfc_cv(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=max(min(max_features, 0.999), 1e-3),
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(
        f=rfc_crossval,
        pbounds={
            "n_estimators": (10, 250),
            "min_samples_split": (2, 25),
            "max_features": (2, 50),
        },
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=10)

    print("Final result:", optimizer.max)
    


# COMMAND ----------

# Optimizing Support Vector Machine
print(Colours.yellow("--- Optimizing SVM ---"))
optimize_svc(data, targets)


# COMMAND ----------

# Random Forest
print(Colours.green("--- Optimizing Random Forest ---"))
optimize_rfc(data, targets)

# COMMAND ----------

# Train Random Forest with optimum parameters
optimize
#params_rfc ={'max_features':int(optimize_rfc['params'].get('max_features')),
#  'min_samples_split':int(optimize_rfc.max['params'].get('min_samples_split')),
#  'n_estimators':int(optimize_rfc.max['params'].get('n_estimators'))
#          }
#params_rfc

# COMMAND ----------

# MAGIC %md
# MAGIC #### Gradient Boosting - xgboost

# COMMAND ----------

# Encoding for Xgboost
from sklearn import preprocessing
import xgboost as xgb

le=preprocessing.LabelEncoder()
le.fit(labels)
dataset['categorical_label']=le.transform(labels)

labels=dataset['categorical_label']


#Converting the dataframe into XGBoost’s Dmatrix object
dtrain=xgb.DMatrix(data, label=labels)

print(dtrain.feature_names)
print(dtrain.get_label())


# COMMAND ----------

#Bayesian Optimization function for xgboost
#specify the parameters you want to tune as keyword arguments
def bo_tune_xgb(max_depth, gamma, n_estimators, learning_rate):
  params= {'max_depth': int(max_depth),
            'gamma': gamma,
            #'booster': 'gbtree',
            #'n_estimators': int(n_estimators),
            #'early_stopping_rounds': 10,
            'learning_rate':learning_rate,
            'subsample': 0.8,
            'eta': 0.1,
            #'eps': 1,
            'colsample_bytree': 0.3, 
            'random_state':0, 
            'seed': 1234,
            'missing':None,
            #'sample_type': 'uniform',
            #'normalize_type': 'tree',
            #'rate_drop': 0.1,
            'objective': 'binary:logistic',
            #'objective': 'binary:hinge',
            #'metric': 'binary_logloss'}
            #'objective':'multi:softprob',  # Multiclass
            'eval_metric': 'logloss'} # 'eval_metric': 'mlogloss' ved flere klasser

# Cross validating with the specified parameters in 5 folds and 70 iterations
  cv_result = xgb.cv(params = params, 
                     dtrain = dtrain, 
                     num_boost_round = 70, 
                     nfold = 5, 
                     early_stopping_rounds = 10, 
                     as_pandas = True)  # we will get the result as a pandas DataFrame.

# Return the log_loss
  return -1.0 * cv_result['train-logloss-mean'].iloc[-1] #


#Invoking the Bayesian Optimizer with the specified parameters to tune
xgb_bo = BayesianOptimization(bo_tune_xgb, {'max_depth': (3, 10),
                                            'gamma': (0, 1),
                                            'learning_rate':(0, 1),
                                            'n_estimators':(100, 200)})

# Performing Bayesian optimization for 5 iterations with 8 steps of random exploration with an #acquisition function of expected improvement
xgb_bo.maximize(n_iter=5, init_points=8, acq='ucb') # Acquisition function. ucb = Upper Confidence Bound. ei



# COMMAND ----------

print("Final result - optimal parameters:", xgb_bo.max)
print("params: ", xgb_bo.max['params'])

params_xgb={'gamma':int(xgb_bo.max['params'].get('gamma')),
  'learning_rate':int(xgb_bo.max['params'].get('learning_rate'))
  #'max_dept':int(xgb_bo.max['params'].get('max_depth')),
  #'n_estimators':int(xgb_bo.max['params'].get('n_estimators'))
          }
params_xgb

# COMMAND ----------

# Train model with found parameters
model = xgb.train(params=params_xgb, 
                  dtrain=dtrain,
        verbose_eval=10)


# COMMAND ----------

# Performance on testdataset.
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

preds = model.predict(dtrain) # Bedre med opsplitning af train og test.
best_preds = np.asarray([np.argmax(line) for line in preds])

print("Precision = {}".format(precision_score(labels, best_preds, average='macro')))
print("Recall = {}".format(recall_score(labels, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(labels, best_preds)))



# COMMAND ----------

# MAGIC %md
# MAGIC #### Confusion Matrix
# MAGIC #### ROC-Curve
# MAGIC #### Save og Load model
# MAGIC #### Predict på "nye data"

# COMMAND ----------

# Feature Importance
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 12))
xgb.plot_importance(model)
plt.show()

# COMMAND ----------

# Regression
def objective(self, max_depth, eta, max_delta_step, colsample_bytree, subsample):
    cur_params =  {'objective': 'reg:linear',
                   'max_depth': int(max_depth),
                   'eta': eta,
                   'max_delta_step': int(max_delta_step),
                   'colsample_bytree': colsample_bytree,
                   'subsample': subsample}

    cv_results = xgb.cv(params=cur_params, 
                        dtrain=self.dm_input, 
                        nfold=3, 
                        seed=3,
                        num_boost_round=50000,
                        early_stopping_rounds=50,
                        metrics='rmse')

    return -1 * cv_results['test-rmse-mean'].min()

# COMMAND ----------

# Class 
class custom_bayesopt:
    def __init__(self, dm_input):
        self.dm_input = dm_input
        
    def objective(self, max_depth, eta, max_delta_step, colsample_bytree, subsample):
        cur_params =  {'objective': 'reg:squarederror',
                       'max_depth': int(max_depth),
                       'eta': eta,
                       'max_delta_step': int(max_delta_step),
                       'colsample_bytree': colsample_bytree,
                       'subsample': subsample}

        cv_results = xgb.cv(params=cur_params, 
                            dtrain=self.dm_input, 
                            nfold=3, 
                            seed=3,
                            num_boost_round=50000,
                            early_stopping_rounds=50,
                            metrics='rmse')

        return -1 * cv_results['test-rmse-mean'].min()

# COMMAND ----------

bopt_process = bopt.BayesianOptimization(custom_bayesopt(dm_input).objective, 
                                         {'max_depth': (2, 15),
                                          'eta': (0.01, 0.3),
                                          'max_delta_step': (0, 10),
                                          'colsample_bytree': (0, 1),
                                          'subsample': (0, 1)},
                              random_state=np.random.RandomState(1))

# COMMAND ----------

# Executing code
bopt_process.maximize(n_iter=10, init_points=12)

# COMMAND ----------

# Winning model parameters:
bopt_process.max

# COMMAND ----------

from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
import xgboost as xgb
import numpy

def xgbCv(train, features, numRounds, eta, gamma, maxDepth, minChildWeight, subsample, colSample):
    # prepare xgb parameters 
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "tree_method": 'auto',
        "silent": 1,
        "eta": eta,
        "max_depth": int(maxDepth),
        "min_child_weight" : minChildWeight,
        "subsample": subsample,
        "colsample_bytree": colSample,
        "gamma": gamma
    }
    cvScore = kFoldValidation(train, features, params, int(numRounds), nFolds = 3)
    print('CV score: {:.6f}'.format(cvScore))
    return -1.0 * cvScore   # invert the cv score to let bayopt maximize
   
def bayesOpt(train, features):
    ranges = {
        'numRounds': (1000, 5000),
        'eta': (0.001, 0.3),
        'gamma': (0, 25),
        'maxDepth': (1, 10),
        'minChildWeight': (0, 10),
        'subsample': (0, 1),
        'colSample': (0, 1)
    }
    # proxy through a lambda to be able to pass train and features
    optFunc = lambda numRounds, eta, gamma, maxDepth, minChildWeight, subsample, colSample: xgbCv(train, features, numRounds, eta, gamma, maxDepth, minChildWeight, subsample, colSample)
    bo = BayesianOptimization(optFunc, ranges)
    bo.maximize(init_points = 50, n_iter = 5, kappa = 2, acq = "ei", xi = 0.0)
    
    bestAUC = round((-1.0 * bo.res['max']['max_val']), 6)
    print("\n Best AUC found: %f" % bestAUC)
    print("\n Parameters: %s" % bo.res['max']['max_params'])
    

def kFoldValidation(train, features, xgbParams, numRounds, nFolds, target='is_pass'):
    kf = KFold(len(train), n_folds = nFolds, shuffle = True)
    fold_score=[]
    
    for train_index, cv_index in kf:
        # split train/validation
        X_train, X_valid = train[features].as_matrix()[train_index], train[features].as_matrix()[cv_index]
        y_train, y_valid = train[target].as_matrix()[train_index], train[target].as_matrix()[cv_index]
        dtrain = xgb.DMatrix(X_train, y_train) 
        dvalid = xgb.DMatrix(X_valid, y_valid)
        
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        gbm = xgb.train(xgbParams, dtrain, numRounds, evals = watchlist, early_stopping_rounds = 100)
        
        score = gbm.best_score
        fold_score.append(score)
    
    return numpy.mean(fold_score)

# COMMAND ----------

print(Colours.green("--- Optimizing Xgboost  ---"))
kFoldValidation(data, targets)

# COMMAND ----------


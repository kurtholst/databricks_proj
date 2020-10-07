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
# MAGIC 5. Train ML models
# MAGIC 6. Evaluate ML performance
# MAGIC 7. Confusion Matrix
# MAGIC 8. Save Model
# MAGIC 9. Load Model
# MAGIC 10. Predict on new data

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

df.write.format("parquet").mode("overwrite").saveAsTable(permanent_table_name)

# COMMAND ----------

# Import Libraries.

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
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

# Split-out validation dataset i data og target klasse.
array = dataset.values
data = array[:,0:60].astype(float)
targets = array[:,60]

# For simplificering:
X=data
y=targets

#####################################
# Split i training og test dataset. #
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                    train_size=0.5,
                                                    test_size=0.5,
                                                    random_state=122)
print("Labels for training and testing data")
print(train_y)
print(test_y)


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
    optimizer_max = optimizer.max
    return optimizer_max

# COMMAND ----------

# Optimizing Support Vector Machine
# Run bayesian optimization function with first batch of iterations
print(Colours.yellow("--- Optimizing SVM ---"))
optimize_svc(data, targets)


# COMMAND ----------

###################### Bayesian Optimization ###########
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
    estimator = RandomForestClassifier(
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

# Random Forest Execute baysian optimization run
print(Colours.green("--- Optimizing Random Forest ---"))
optimize_rfc(data, targets)

# COMMAND ----------

# Train Random Forest with optimum parameters found during bayesian optimization
# Fiske optimale parametre ud af optimize_rfc
trainedforest = RandomForestClassifier(max_features = 44, 
                                       min_samples_split = 10, 
                                       n_estimators = 130) \
                    .fit(train_X, train_y)

trainedforest

# COMMAND ----------

# Plotting Learning Curve
# Create CV training and test scores for various training set sizes
from sklearn.model_selection import learning_curve

X = data
y = targets

train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(),
                                                        X, 
                                                        y,
                                                        # Number of folds in cross-validation
                                                        cv = 10,
                                                        # Evaluation metric
                                                        scoring = 'accuracy',
                                                        # Use all computer cores
                                                        n_jobs = -1,
                                                        # 50 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 50))

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)



# COMMAND ----------

# Confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
y_pred = trainedforest.predict(test_X)

#train_X, test_X, train_y, test_y 
# Machine Learning Performance
print("Confusion Matrix")
print(confusion_matrix(y_true = test_y, y_pred = y_pred))


# COMMAND ----------

# Classification Report (ala caret)
from sklearn.metrics import classification_report
print("Classification Report")
print(classification_report(y_true = test_y, y_pred = y_pred))


# COMMAND ----------

# Accuracy
acc5 = accuracy_score(y_true = test_y, y_pred = y_pred)
print(acc5)

# COMMAND ----------

# calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(test_y, y_pred))

# COMMAND ----------

# store the predicted probabilities for class 1
classifier = trainedforest.fit(X, y)
predictions = classifier.predict_proba(test_X)
print(predictions)
#y_pred_prob = RFC.predict_proba(data)


# COMMAND ----------

# RoC-Curve
# IMPORTANT: first argument is true values, second argument is predicted probabilities

# we pass y_test and y_pred_prob
# we do not use y_pred_class, because it will give incorrect results without generating an error
# roc_curve returns 3 objects fpr, tpr, thresholds
# fpr: false positive rate
# tpr: true positive rate
fpr, tpr, thresholds = metrics.roc_curve(test_y, predictions)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Gradient Boosting - xgboost

# COMMAND ----------

# Encoding for Xgboost
from sklearn import preprocessing
import xgboost as xgb

le=preprocessing.LabelEncoder()
labels = targets

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
                     early_stopping_rounds = 10 , 
                     as_pandas = True)  # we will get the result as a pandas DataFrame.

# Return the log_loss
  return -1.0 * cv_result['train-logloss-mean'].iloc[-1] #


#Invoking the Bayesian Optimizer with the specified parameters to tune
xgb_bo = BayesianOptimization(bo_tune_xgb, {'max_depth': (3, 10),
                                            'gamma': (0, 1),
                                            'learning_rate':(0.01, 1),
                                            'n_estimators':(100, 200)})




# COMMAND ----------

# Performing Bayesian optimization for 5 iterations with 8 steps of random exploration with an #acquisition function of expected improvement
xgb_bo.maximize(n_iter=5, init_points=8, acq='ucb') # Acquisition function. ucb = Upper Confidence Bound. Other alternatives: ei


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

#

bopt_process = bopt.BayesianOptimization(custom_bayesopt(dm_input).objective, 
                                         {'max_depth': (2, 15),
                                          'eta': (0.01, 0.3),
                                          'max_delta_step': (0, 10),
                                          'colsample_bytree': (0, 1),
                                          'subsample': (0, 1)},
                              random_state=np.random.RandomState(1))

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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


fig, axes = plt.subplots(3, 2, figsize=(10, 15))

X, y = load_digits(return_X_y=True)

title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Catboost

# COMMAND ----------

# Supervised Learning med 
# Catboost (Yandex)

# Categorical features
#import numpy as np
#categorical_features = [col for c, col in enumerate(dataset.columns)
#                        if not (np.issubdtype(dataset.dtypes[c], np.number))]
#print(categorical_features)
from catboost import CatBoostClassifier, Pool
import ipywidgets # Nødvendig for plot

cb_model = CatBoostClassifier(iterations=100,
                              loss_function='Logloss',
                              #cat_features=categorical_features,
                              verbose=True) #, task_type = "GPU"



# COMMAND ----------

# Training model
cb_model.fit(X = data, 
             y = targets, 
             use_best_model = True, 
             #cat_features=categorical_features,
             plot = True)

# COMMAND ----------

# make the prediction using the resulting model
preds_class = cb_model.predict(data, prediction_type='Class')
preds_raw_vals = cb_model.predict(data, prediction_type='RawFormulaVal')
preds_proba = cb_model.predict(data, prediction_type='Probability')

# Eller preds_proba = cb_model.predict_proba(data)
print("class = ", preds_class)
print("proba = ", preds_proba)
print("proba = ", preds_raw_vals)

# COMMAND ----------

# MAGIC %md
# MAGIC # Bayesian Optimization using skopt

# COMMAND ----------

# Model selection
# Reference: https://www.kaggle.com/shivampanwar/catboost-and-hyperparameter-tuning-using-bayes
from sklearn.model_selection import StratifiedKFold

# Metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer

# Skopt functions
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, VerboseCallback, DeltaXStopper
from skopt.space import Real, Categorical, Integer
from time import time

# COMMAND ----------

# Reporting util for different optimizers
def report_perf(optimizer, X, y, title, callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers
    
    optimizer = a sklearn or a skopt optimizer
    X = the training set 
    y = our target
    title = a string label for the experiment
    """
    import pandas as pd
    import pprint
    start = time()
    if callbacks:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)
    d=pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_
    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
           +u"\u00B1"+" %.3f") % (time() - start, 
                                  len(optimizer.cv_results_['params']),
                                  best_score,
                                  best_score_std))    
    print('Best parameters:')
    pprint.pprint(best_params)
    print()
    return best_params

roc_auc = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# COMMAND ----------

clf = CatBoostClassifier(thread_count=2,
                         loss_function='Logloss',
                         od_type = 'Iter',
                         verbose= True)

# COMMAND ----------

# Defining Catboost search space
search_spaces = {'iterations': (10, 1000),
                 'depth': (1, 8),
                 #'learning_rate': (0.01, 1.0, 'log-uniform'),
                 'learning_rate': (0.01, 1.0),
                 #'random_strength': (1e-9, 10, 'log-uniform'),
                 'random_strength': (1e-9, 10),
                 'bagging_temperature': (0.0, 1.0),
                 'border_count': (1, 255),
                 'l2_leaf_reg': (2, 30),
                 #'scale_pos_weight':(0.01, 1.0, 'uniform')}
                 'scale_pos_weight':(0.01, 1.0)}

# COMMAND ----------

# Setting up BayesSearchCV
opt = BayesSearchCV(estimator=clf, 
                    search_spaces=search_spaces,
                    scoring=roc_auc,
                    cv=skf,
                    n_iter=10,
                    n_jobs=1,  # use just 1 job with CatBoost in order to avoid segmentation fault
                    return_train_score=False,
                    refit=True,
                    optimizer_kwargs={'base_estimator': 'GP'},
                    random_state=42)

# COMMAND ----------

# Execute bayesian 
best_params = report_perf(optimizer = opt, 
                          X = X, 
                          y = y, 
                          title = 'CatBoost', 
                          callbacks=[VerboseCallback(100), DeadlineStopper(60*10)])


# COMMAND ----------

# Convert ordered dictionary to dictionary
import json
best_params_bayesian = json.loads(json.dumps(best_params))


# COMMAND ----------

# Manuelt
best_params={'bagging_temperature': 0.41010395885331385,
 'border_count': 186,
 'depth': 8,
 'iterations': 323,
 'l2_leaf_reg': 21,
 'learning_rate': 0.0673344419215237,
 'random_strength': 3.230824361824754e-06,
 'scale_pos_weight': 0.7421091918485163}

best_params['iterations']=100
type(best_params)

# COMMAND ----------

# MAGIC %%time
# MAGIC tuned_model = CatBoostClassifier(**best_params_bayesian, # **best_params, 
# MAGIC                                  task_type = "CPU", 
# MAGIC                                  od_type='Iter', 
# MAGIC                                  one_hot_max_size=10)
# MAGIC tuned_model.fit(X = data, 
# MAGIC              y = targets)

# COMMAND ----------

# make the prediction using the resulting model
preds_class = tuned_model.predict(data, prediction_type='Class')
preds_raw_vals = tuned_model.predict(data, prediction_type='RawFormulaVal')
preds_proba = tuned_model.predict(data, prediction_type='Probability')

# Eller preds_proba = cb_model.predict_proba(data)
print("class = ", preds_class)
print("proba = ", preds_proba)
print("proba = ", preds_raw_vals)

# COMMAND ----------

# Let's look at a confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true = y, y_pred = preds_class)


# COMMAND ----------

data.

# COMMAND ----------

  
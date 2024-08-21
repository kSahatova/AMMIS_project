import warnings
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

import seaborn as sns

from utils import read_data, bin_feature
from utils import numeric_feature_preprocessor, categorical_feature_preprocessor
from utils import make_generic_pipeline, set_features_names, remove_correlated_features
from utils import calc_auc


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)

pd.set_option('display.max_columns', 50)
sns.set_theme()

# Reading data

data_folder = 'C:\\Users\\sahat\\OneDrive - KU Leuven\\Documents\\AA2024\\Assignment1\\data'
X, y = read_data(data_folder, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train['Age'], train_bins = bin_feature(X_train, 'Age')
X_test['Age'], _ = bin_feature(X_test, 'Age', bins=train_bins)

X_train = remove_correlated_features(X_train)
X_test = remove_correlated_features(X_test)

# Preprocessing
numeric_columns = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
oh_enc_columns = ['Age', 'Gender', 'high Dropped calls', 'No Usage', 'Tariff_OK']
count_enc_columns = ['Handset', 'tariff']
ord_enc_columns = ['Usage_Band']

categories = [['Low', 'MedLow', 'Med', 'MedHigh', 'High']]

num_preprocessor = numeric_feature_preprocessor(numeric_columns)
cat_preprocessor = categorical_feature_preprocessor(ohe_columns=oh_enc_columns,
                                                    count_enc_columns=count_enc_columns,
                                                    ord_enc_columns=ord_enc_columns,
                                                    ordered_categories=categories)

pipeline = make_generic_pipeline(num_preprocessor, cat_preprocessor)
X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(X_test)

X_train_transformed = set_features_names(pipeline, X_train_transformed)
X_test_transformed = set_features_names(pipeline, X_test_transformed)

# Save transformed data
dump(X_train_transformed, 'processed_data/telco_churn/x_train.joblib')
dump(X_test_transformed, 'processed_data/telco_churn/x_test.joblib')

dump(y_train, 'processed_data/telco_churn/y_train.joblib')
dump(y_test, 'processed_data/telco_churn/y_test.joblib')

# Training

neg, pos = y_train.value_counts()
scale_pos_weight = np.round(neg / pos)

# GridSearch for selection of the optimal parameters
'''
clf = XGBClassifier()
params = {'n_estimators': [100, 200, 300],
          'max_depth': [6, 8, 10],
          'colsample_bytree': [0.7, 1],
          'scale_pos_weight': [scale_pos_weight],
          'eval_metric': ['logloss'],
          'objective': ['binary:logistic']}

gs = GridSearchCV(clf, param_grid=params, cv=5, scoring='roc_auc')
gs.fit(X_train_transformed, y_train)

xgboost_optimal = gs.best_estimator_
print('Best score: ', gs.best_score_)
print('Best parameters: ', gs.best_params_)

'''

# Train the pretrained_models
best_params = {'colsample_bytree': 1, 'eval_metric': 'logloss',
               'max_depth': 10, 'n_estimators': 200,
               'objective': 'binary:logistic', 'scale_pos_weight': 6.0}
xgboost_optimal = XGBClassifier(**best_params)
xgboost_optimal.fit(X_train_transformed, y_train)

# Check AUC on the test set
calc_auc(xgboost_optimal, X_test_transformed, y_test)

# Save the trained pretrained_models
dump(xgboost_optimal, 'pretrained_models/xgboost_telcochurn.joblib')




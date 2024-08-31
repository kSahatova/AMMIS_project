import shap
import joblib
import numpy as np
import pandas as pd
import os.path as osp


aggregated_concepts = {
    'Shipment': ['Type', 'Days for shipment (scheduled)', 'Shipping Mode', 'Distance (km)'],
    'Customer': ['Customer Zipcode', 'Customer Segment'],
    'Department': ['Department Name', 'Market'],
    'Store': ['Store Latitude', 'Store Longitude'],
    'Order': ['Order Id', 'Order Longitude', 'Order Latitude', # 'order date (DateOrders)', 
              'Benefit per order', 'Order Item Total', 'Order Status', 'Sales', 
              'Order Item Discount', 'order_year', 'order_month', 'order_day'],
    'Product': ['Category Name']
}


# 'Product Status' all values are 0
# 'Order Profit Per Order' is equal to 'Benefit per order'
# 'Order Item Total' is equal to the feature 'Sales per customer'


# loading test data
main_dir = r'D:\PycharmProjects\AMMISproject'
processed_data_dir = 'data\processed_data'
dataset = 'dataco'
X_test = joblib.load(osp.join(main_dir, processed_data_dir, dataset, 'x_test_std.joblib'))
y_test = joblib.load(osp.join(main_dir, processed_data_dir, dataset, 'y_test.joblib'))

# loading a pretrained model
pretrained_models_dir = 'SuperImposition\pretrained_models'
model_name = 'xgboost' 
model = joblib.load(osp.join(main_dir, pretrained_models_dir, f'{model_name}_dataco.joblib'))

# Calculate marginal contribution

# Specify whether to use precalculated shap values or compute them (takes long time, i.e. mlp ~ 1h)
load_shap_values = True
outputs_dir = r'D:\\PycharmProjects\\AMMISproject\\SuperImposition\\outputs'

if load_shap_values:
    shap_values_df = pd.read_csv(osp.join(outputs_dir, f'shap_values_{model_name}_{dataset}.csv'), index_col=0)    

else:     
    if model_name in ['xgboost', 'dt']:
        explainer = shap.TreeExplainer(model)
    elif model_name == 'mlp':  # use std test set 
        explainer = shap.Explainer(model.predict, masker=shap.maskers.Partition(data=X_test, clustering='correlation'))

    shap_values = explainer.shap_values(X_test)

    if len(X_test.shape) != len(shap_values.shape):
        shap_values = shap_values[:, :, 1]

    shap_values_df = pd.DataFrame(shap_values, columns=X_test.columns)
    shap_values_df.to_csv(osp.join(outputs_dir, f'shap_values_{model_name}_dataco.csv'))


# Explore derived shap_values

print(shap_values_df.shape)
print('The shape matches the shape of the test set :', 
      shap_values_df.shape[0] == X_test.shape[0])
print(shap_values_df.isnull().sum(axis=0))


# 1) Calculation of feature contributions
feature_contribs = shap_values_df * X_test

# a) SHAP values for categorical features are normally summed. Thus, we summ of categorical features that were one hot encoded
ohe_columns = ['Type', 'Customer Segment', 'Market', 'Shipping Mode', 'Order Status']

# b) calculating mean value of all features contributions
# # creating a dataframe where the one-hot encoded features are sumed, and the rest are left unchanged
agg_features_contributions = feature_contribs.copy(deep=True) 

for val in ohe_columns:
    categories = []
    for col in X_test.columns:
        if val in col:
            categories.append(col)
    
    agg_features_contributions[val] = feature_contribs.loc[:, categories].sum(axis=1)
    agg_features_contributions.drop(categories, axis=1, inplace=True)

feature_contribs_avg = agg_features_contributions.mean(axis=0)


# 2) calculating concept contributions
concepts_contributions = pd.DataFrame(0, 
                                      columns=list(aggregated_concepts.keys()),
                                      index=np.arange(1)
)
for concept, features in aggregated_concepts.items():
    concepts_contributions[concept] = feature_contribs_avg.loc[features].sum()

concepts_contributions.to_csv(osp.join(outputs_dir, f'concept_contributions_{model_name}_dataco.csv'))
print(concepts_contributions)
import shap
import joblib
import pandas as pd
import os.path as osp

import seaborn as sns
from IPython.display import HTML, display
from sklearn.preprocessing import MinMaxScaler

"""
# THE WRONG ONE
aggregated_concepts = {
    'Shipment': ['Type', 'Days for shipment (scheduled)', 'Shipping Mode', 'Distance (km)'],
    'Customer': ['Customer Zipcode', 'Customer Segment'],
    'Department': ['Department Name', 'Market'],
    'Store': ['Store Latitude', 'Store Longitude'],
    'Order': ['Order Id', 'Order City', 'Order Country', 'order date (DateOrders)',
              'Order Profit Per Order', 'Order Status', 'Sales', 'Order Item Discount',
              'order_year', 'order_month', 'order_day'],
    'ProductCategory': ['Category Name']
}
"""


aggregated_concepts = {
    'Shipment': ['Type', 'Days for shipment (scheduled)', 'Shipping Mode', 'Distance (km)'],
    'Customer': ['Customer Zipcode', 'Customer Segment'],
    'Department': ['Department Name', 'Market'],
    'Store': ['Store Latitude', 'Store Longitude'],
    'Order': ['Order Id', 'Order Longitude', 'Order Latitude',
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
model_name = 'mlp' 
model = joblib.load(osp.join(main_dir, pretrained_models_dir, f'{model_name}_dataco.joblib'))

# Calculate marginal contribution

load_shap_values = False
shap_val_dir = r'D:\\PycharmProjects\\AMMISproject\\SuperImposition\\outputs'

if load_shap_values:
    shap_values = pd.read_csv(osp.join(shap_val_dir, f'shap_values_{model_name}_{dataset}.csv'), index_col=0)    

else:     
    if model_name in ['xgboost', 'dt']:
        explainer = shap.TreeExplainer(model)
    elif model_name == 'mlp':  # use std test set 
        explainer = shap.Explainer(model.predict, masker=shap.maskers.Partition(data=X_test, clustering='correlation'))

    shap_values = explainer.shap_values(X_test)

    if len(X_test.shape) != len(shap_values.shape):
        shap_values = shap_values[:, :, 1]

    shap_values_df = pd.DataFrame(shap_values, columns=X_test.columns)
    shap_values_df.to_csv(osp.join(shap_val_dir, f'shap_values_{model_name}_dataco.csv'))


# In the preprocessed data the naming of the columns differs, so we have to define
# expanded features and put them as values of corresponding concepts
concepts_attributes = {}
for concept, features in aggregated_concepts.items():
    extended_features = []
    for value in features:
            [extended_features.append(column) for column in X_test.columns if value in column]
   
    concepts_attributes[concept] = extended_features

# 0s and 1s for one-hot encoded and consequently scaled features are mapped to the same values, e.g. -0.0123 for all 0s in a column and 0.0432 for all 1s

feature_contribs = shap_values * X_test

# SHAP values for categorical features are normally summed. Thus, we summ of categorical features that were one hot encoded
ohe_columns = ['Type', 'Customer Segment', 'Market', 'Shipping Mode', 'Order Status']
# creating a dataframe where the one-hot encoded features are sumed, and the rest are left unchanged
agg_features_contributions = feature_contribs.copy(deep=True)

for val in ohe_columns:
    categories = []
    for col in X_test.columns:
        if val in col:
            categories.append(col)
    
    agg_features_contributions[val] = feature_contribs.loc[:, categories].sum(axis=1)
    agg_features_contributions.drop(categories, axis=1, inplace=True)



# Calculate concept contribution

# a) normalizing features contributions -> calculating mean of the features contributions
features_contribs_avg = agg_features_contributions.mean(axis=0)

"""
# b) calculating mean(g_c) - scaled mean for all feature contributions
# calculation of the aggregated 
concepts = list(concepts_attributes.keys())
scaled_mean_fcs = pd.DataFrame(0, columns=concepts, index=feature_contribs_normalized.index)

for concept, features_names in concepts_attributes.items():
    fcs = feature_contribs_normalized.loc[:, features_names]
    scaled_mean_fcs[concept] = fcs.mean(axis=1)"""

# c) calculating concept contributions
concepts_contributions = pd.DataFrame(columns=[e for v in aggregated_concepts.values() for e in v])
for concept, features in aggregated_concepts.items():
    for value in features:
        combined_features = []
        [combined_features.append(column) for column in X_test.columns if value in column]
        # we sum the one-hot encoded features
        concepts_contributions[value] = features_contribs_avg.loc[:, combined_features].sum(axis=1)

# order_date_columns = ['order_year', 'order_month', 'order_day']
# combined_features_contribution['order date (DateOrders)'] = \
#     combined_features_contribution.loc[:, order_date_columns].sum(axis=1)
# combined_features_contribution.drop(order_date_columns, axis=1, inplace=True)


cm = sns.light_palette("green", as_cmap=True)
s = combined_features_contribution.style.background_gradient(cmap=cm)
print(s)


combined_f_html = combined_features_contribution.to_html()
with open(f"combined_features_contribution_{model_name}.html", "w") as f:
    f.write(combined_f_html)

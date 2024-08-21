import shap
import joblib
import pandas as pd
import os.path as osp

import seaborn as sns
from IPython.display import HTML, display
from sklearn.preprocessing import MinMaxScaler


# loading test data
main_dir = r'D:\PycharmProjects\AMMISproject\SuperImposition'
processed_data_dir = 'processed_data'
dataset = 'dataco'
X_test = joblib.load(osp.join(main_dir, processed_data_dir, dataset, 'x_test.joblib'))
y_test = joblib.load(osp.join(main_dir, processed_data_dir, dataset, 'y_test.joblib'))

# loading a pretrained model
pretrained_models_dir = 'pretrained_models'
model = joblib.load(osp.join(main_dir, pretrained_models_dir, 'dt_dataco.joblib'))

# Calculate marginal contribution
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# shap.summary_plot(shap_values, X_test)
# shap.summary_plot(shap_values, X_test, plot_type="bar")

# Calculate feature contribution
# Concepts for the telco_churn dataset
# concepts_attributes = {'Customer': ['Age_group1', 'Age_group2',
#                                     'Age_group3', 'Age_group4',
#                                     'Age_group5', 'Gender_F',
#                                     'Gender_M'],
#                        'Subscription': ['Tariff_OK_High CAT 100',
#                                         'Tariff_OK_High CAT 50',
#                                         'Tariff_OK_High Play 100',
#                                         'Tariff_OK_OK', 'tariff',
#                                         'Usage_Band'],
#                        'CustomerSetup': ['Handset'],
#                        'SubscriptionHistory': ['Dropped_Calls', 'Peak_mins_Sum',
#                                                'OffPeak_mins_Sum', 'Weekend_mins_Sum',
#                                                'International_mins_Sum', 'National mins',
#                                                'All_calls_mins']}


aggregated_concepts = {
    'Shipment': ['Type', 'Days for shipment (scheduled)', 'Shipping Mode'],
    'Customer': ['Customer Zipcode', 'Customer Segment'],
    'Department': ['Department Name', 'Market'],
    'Store': ['Latitude', 'Longitude'],
    'Order': ['Order Id', 'Order City', 'Order Country', 'order date (DateOrders)',
              'Order Profit Per Order', 'Order Status', 'Sales', 'Order Item Discount',
              'order_year', 'order_month', 'order_day'],
    'ProductCategory': ['Category Name']
}

# In the preprocessed data the naming of the columns differs, so we have to define
# expanded features and put them as a value of a corresponding concept
concepts_attributes = {}
for concept, features in aggregated_concepts.items():
    extended_features = []
    for value in features:
        if value in X_test.columns:
            [extended_features.append(column) for column in X_test.columns if value in column]
    concepts_attributes[concept] = extended_features

if len(X_test.shape) != len(shap_values.shape):
    shap_values = shap_values[:, :, 1]
feature_contribs = shap_values * X_test


# Calculate concept contribution

# a) scaling feature contributions
scaler = MinMaxScaler()
fcs_normalized = scaler.fit_transform(feature_contribs)
fcs_normalized = pd.DataFrame(fcs_normalized, columns=feature_contribs.columns)

# b) calculating mean(g_c) - scaled mean for all feature contributions
concepts = list(concepts_attributes.keys())
scaled_mean_fcs = pd.DataFrame(0, columns=concepts, index=fcs_normalized.index)

for concept, features_names in concepts_attributes.items():
    fcs = fcs_normalized.loc[:, features_names]
    scaled_mean_fcs[concept] = fcs.mean(axis=1)

# c) calculating concept contributions
combined_features_contribution = pd.DataFrame(columns=[e for v in aggregated_concepts.values() for e in v])
for concept, features in aggregated_concepts.items():
    for value in features:
        combined_features = []
        [combined_features.append(column) for column in X_test.columns if value in column]
        # we sum the one-hot encoded features
        combined_feature_contribution = fcs_normalized.loc[:, combined_features].sum(axis=1)
        combined_features_contribution[value] = combined_feature_contribution

order_date_columns = ['order_year', 'order_month', 'order_day']
combined_features_contribution['order date (DateOrders)'] = \
    combined_features_contribution.loc[:, order_date_columns].sum(axis=1)
combined_features_contribution.drop(order_date_columns, axis=1, inplace=True)

concept_contributions = scaled_mean_fcs.sum(axis=1)
concept_contributions.rename('concept_contribution', inplace=True)

cm = sns.light_palette("green", as_cmap=True)
combined_features_contribution.style.background_gradient(cmap=cm)
display(combined_features_contribution)

combined_f_html = combined_features_contribution.to_html()
with open("combined_features_contribution.html", "w") as f:
    f.write(combined_f_html)

with open("scaled_mean_fcs.html", "w") as f:
    f.write(html)
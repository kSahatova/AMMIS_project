import shap
import joblib
import pandas as pd
import os.path as osp

from sklearn.preprocessing import MinMaxScaler


processed_data_dir = 'processed_telco_churn_data'
X_train = joblib.load(osp.join(processed_data_dir, 'X_train.joblib'))
X_test = joblib.load(osp.join(processed_data_dir, 'X_test.joblib'))
y_train = joblib.load(osp.join(processed_data_dir, 'y_train.joblib'))
y_test = joblib.load(osp.join(processed_data_dir, 'y_test.joblib'))

model = joblib.load('model/fitted_xgboost.joblib')

# Calculate marginal contribution

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# shap.summary_plot(shap_values, X_test)
# shap.summary_plot(shap_values, X_test, plot_type="bar")


# Calculate feature contribution
# todo: Preliminary concepts (entities) of the conceptual model that were distinguished
concepts_attributes = {'Customer': ['Age_group1', 'Age_group2',
                                    'Age_group3', 'Age_group4',
                                    'Age_group5', 'Gender_F',
                                    'Gender_M'],
                       'Subscription': ['Tariff_OK_High CAT 100',
                                        'Tariff_OK_High CAT 50',
                                        'Tariff_OK_High Play 100',
                                        'Tariff_OK_OK', 'tariff',
                                        'Usage_Band'],
                       'CustomerSetup': ['Handset'],
                       'SubscriptionHistory': ['Dropped_Calls', 'Peak_mins_Sum',
                                               'OffPeak_mins_Sum', 'Weekend_mins_Sum',
                                               'International_mins_Sum', 'National mins',
                                               'All_calls_mins']}

feature_contribs = shap_values * X_test

# Calculate concept contribution

# a) scaling feature contributions
scaler = MinMaxScaler()
fcs_normalized = scaler.fit_transform(feature_contribs)
fcs_normalized = pd.DataFrame(fcs_normalized,
                              columns=feature_contribs.columns)

# b) calculating mean(g_c) - scaled mean for all feature contributions
concepts = list(concepts_attributes.keys())
scaled_mean_fcs = pd.DataFrame(0, columns=concepts, index=fcs_normalized.index)

for concept, features_names in concepts_attributes.items():
    fcs = fcs_normalized.loc[:, features_names]
    scaled_mean_fcs[concept] = fcs.mean(axis=1)

# c) calculating concept contributions
concept_contributions = scaled_mean_fcs.sum(axis=1)
concept_contributions.rename('concept_contribution', inplace=True)


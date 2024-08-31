import joblib
import pandas as pd
import os.path as osp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from category_encoders.count import CountEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from xgboost import XGBClassifier


mem_clf_dir = r'D:\\PycharmProjects\\AMMISproject\\SuperImposition\\pretrained_models'

"""# Read the data
data_path = r'D:\\PycharmProjects\\AMMISproject\\data\\raw\\DataCoSmartSupplyChain\\DataCoSupplyChainDataset.csv'
processed_data_dir = r'D:\\PycharmProjects\\AMMISproject\\data\\processed_data\\dataco'
data = pd.read_csv(data_path, encoding='latin-1')


statuses_to_drop = ['CANCELED', 'SUSPECTED_FRAUD']
# Drop rows where late delivery risk is always 0 (when the order status is 'CANCELED' or 'SUSPECTED_FRAUD')
filtered_data = data[~data['Order Status'].isin(statuses_to_drop)]


# 'Product Status' all values are 0, so we don't consider it
# 'Order Profit Per Order' is equal to 'Benefit per order'
# 'Order Item Total' is equal to the feature 'Sales per customer'

selected_columns = ['Type', 'Days for shipment (scheduled)', 'order date (DateOrders)',
                    'Category Name', 'Customer Segment', 'Customer Zipcode',
                    'Department Name', 'Latitude', 'Longitude', 'Market',
                    'Order City', 'Order Country', 'Sales', 'Order Id',
                    'Order Item Discount', 'Order Profit Per Order',
                    'Order Status', 'Shipping Mode', 'Late_delivery_risk']

filtered_data = filtered_data.loc[:, selected_columns]
target_column = 'Late_delivery_risk'
targets = filtered_data.loc[:, target_column]
features = filtered_data.drop(target_column, axis=1)

features = features.rename({'Latitude': 'Store Latitude',
                            'Longitude': 'Store Longitude'})

features['order_date'] = pd.to_datetime(features['order date (DateOrders)'])
features['order_year'] = pd.DatetimeIndex(features['order_date']).year.astype('float64')
features['order_month'] = pd.DatetimeIndex(features['order_date']).month.astype('float64')
features['order_day'] = pd.DatetimeIndex(features['order_date']).day.astype('float64')
features.drop(['order date (DateOrders)', 'order_date'], axis=1, inplace=True)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)

# Depending on the cardinality of the categorical features, we are trying to apply different encoding techniques
count_encoder = CountEncoder(cols=['Category Name', 'Customer Zipcode',
                                   'Department Name', 'Order City', 'Order Country'])
x_train = count_encoder.fit_transform(x_train)
x_test = count_encoder.transform(x_test)

ohe_columns = ['Type', 'Customer Segment', 'Market', 'Shipping Mode', 'Order Status']
x_train = pd.get_dummies(x_train, columns=ohe_columns).reset_index(drop=True)
x_test = pd.get_dummies(x_test, columns=ohe_columns).reset_index(drop=True)

# Standardize the numeric features
scaler = RobustScaler()
preproc_columns = x_train.columns
x_train_std = pd.DataFrame(scaler.fit_transform(x_train), columns=preproc_columns)
x_test_std = pd.DataFrame(scaler.transform(x_test), columns=preproc_columns)

# Save transformed data
joblib.dump(x_train_std, osp.join(processed_data_dir, 'x_train_std.joblib'))
joblib.dump(x_test_std, osp.join(processed_data_dir, 'x_test_std.joblib'))

joblib.dump(x_train_std, osp.join(processed_data_dir, 'x_train.joblib'))
joblib.dump(x_test_std, osp.join(processed_data_dir, 'x_test.joblib'))

joblib.dump(y_train, osp.join(processed_data_dir, 'y_train.joblib'))
joblib.dump(y_test, osp.join(processed_data_dir, 'y_test.joblib'))

corr = x_train.corr()"""


data_dir = 'D:\\PycharmProjects\\AMMISproject\\data\\processed_data'
dataset = 'dataco'

x_train_std = joblib.load(osp.join(data_dir, dataset, 'x_train_std.joblib'))
x_test_std = joblib.load(osp.join(data_dir, dataset, 'x_test_std.joblib'))

x_train = joblib.load(osp.join(data_dir, dataset, 'x_train.joblib'))
x_test = joblib.load(osp.join(data_dir, dataset, 'x_test.joblib'))

y_train = joblib.load(osp.join(data_dir, dataset, 'y_train.joblib'))
y_test = joblib.load(osp.join(data_dir, dataset, 'y_test.joblib'))

print('Shape of the training set: ', x_train_std.shape)
print('Shape of the test set: ', x_test_std.shape)
print('Shape of the trainigb targets: ', y_train.shape)

# Train a simple MLP

# batch_size, architecture, learning rate and activation function : 64, (128, 64), 0.01, 'relu - best MLP model 

model = MLPClassifier(hidden_layer_sizes=(128, 64,), max_iter=500,
                      early_stopping=True, validation_fraction=0.2,
                      batch_size=64, random_state=123)
model.fit(x_train_std, y_train)

# Save the classifier
joblib.dump(model, osp.join(mem_clf_dir, 'mlp_dataco.joblib'))

# Evaluate MLP model
y_prediction = model.predict(x_test_std)
mlp_acc = accuracy_score(y_test, y_prediction)
mlp_roc = roc_auc_score(y_test, y_prediction)
print('Accuracy of the MLP: ', mlp_acc, '\nAUC of the MLP: ', mlp_roc)
print('MLP performance evaluation: \n', classification_report(y_test, y_prediction))

"""
# Check if the pretrained_models captures the pattern where each row is a part of an order which has to be delivered as a whole
y_test = y_test.reset_index(drop=True)
y_pred_df = pd.DataFrame(y_prediction, index=y_test.index, columns=['prediction'])
order_id_check = pd.concat([x_test.loc[:, ['Order Id', 'Days for shipment (scheduled)']],
                            y_test, y_pred_df],
                           axis=1)
order_id_check[order_id_check.duplicated('Order Id', keep=False)].sort_values(by='Order Id')
"""

# Train a decision tree
model = DecisionTreeClassifier(random_state=123)
model.fit(x_train_std, y_train)

# Save the classifier
joblib.dump(model, osp.join(mem_clf_dir, 'dt_dataco.joblib'))


# Evaluate DT model
y_prediction = model.predict(x_test_std)
dt_acc = accuracy_score(y_test, y_prediction)
dt_roc = roc_auc_score(y_test, y_prediction)
print('Accuracy of the DT: ', dt_acc, '\nAUC of the DT: ', dt_roc)
print('DT performance evaluation: \n', classification_report(y_test, y_prediction))


# Train XGBoost model
xgboost_model = XGBClassifier(random_state=123)
xgboost_model.fit(x_train_std, y_train)

# Save the classifier
joblib.dump(xgboost_model, osp.join(mem_clf_dir, 'xgboost_dataco.joblib'))

# Evaluate XGBoost model
y_prediction = xgboost_model.predict(x_test_std)
xgb_acc = accuracy_score(y_test, y_prediction)
xgb_roc = roc_auc_score(y_test, y_prediction)
print('Accuracy of the XGBoost: ', xgb_acc, '\nAUC of the XGBoost: ', xgb_roc)
print('XGBoost performance evaluation: \n', classification_report(y_test, y_prediction))

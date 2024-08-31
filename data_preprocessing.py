import pickle
import pandas as pd
import joblib
import os.path as osp
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from category_encoders.count import CountEncoder
from geopy.distance import great_circle

sns.set_theme(font_scale=0.6)


# Read the data
data_path = r'D:\\PycharmProjects\\AMMISproject\\data\\raw\\DataCoSmartSupplyChain\\DataCoSupplyChainDataset.csv'
processed_data_dir = r'D:\\PycharmProjects\\AMMISproject\\data\\processed_data\\dataco'
data = pd.read_csv(data_path, encoding='latin-1')


statuses_to_drop = ['CANCELED', 'SUSPECTED_FRAUD']
# Drop rows where late delivery risk is always 0 (when the order status is 'CANCELED' or 'SUSPECTED_FRAUD')
filtered_data = data[~data['Order Status'].isin(statuses_to_drop)]


# 'Product Status' all values are 0< therefor, we skip the feature 
# 'Order Profit Per Order' is equal to 'Benefit per order'
# 'Order Item Total' is equal to the feature 'Sales per customer'

# Keeping 'Order City', 'Order Country' out for now
selected_columns = ['Type', 'Days for shipment (scheduled)', 'order date (DateOrders)',
                    'Category Name', 'Customer Segment', 'Customer Zipcode',
                    'Department Name', 'Latitude', 'Longitude', 'Market',
                    'Order City', 'Order Country', 'Sales', 'Order Id',
                    'Order Item Discount', 'Benefit per order', 'Order Item Total',
                    'Order Status', 'Shipping Mode', 'Late_delivery_risk']

filtered_data = filtered_data.loc[:, selected_columns]

# Open city coordinates from a file
with open('PredModels/Data/city_coordinates.pkl', 'rb') as f:
    city_coordinates = pickle.load(f)
city_coordinates = pd.DataFrame(city_coordinates.items(), columns=['Order City', 'Coordinates'])
country_coordinates = pd.read_csv('PredModels/Data/country_coordinates.csv')

filtered_data = pd.merge(filtered_data, city_coordinates, on='Order City', how='left')
# Fill in missing city coordinates
filtered_data['Coordinates'] = filtered_data['Coordinates'].fillna(filtered_data['Order Country'].map(country_coordinates.set_index('Order Country')['Coordinates']))
# Fill in values that are not two numbers (latitude, longitude) with (0, 0) (also replace (None, None) with (0, 0))
filtered_data['Coordinates'] = filtered_data['Coordinates'].apply(lambda x: (0, 0) if not isinstance(x, tuple) or len(x) != 2 else x)
# Fill in values that are contain None (for example (None, None)) with (0, 0)
filtered_data['Coordinates'] = filtered_data['Coordinates'].apply(lambda x: (0, 0) if x[0] is None else x)
# Split the coordinates into latitude and longitude
filtered_data[['Order Latitude', 'Order Longitude']] = pd.DataFrame(filtered_data['Coordinates'].tolist(), index=filtered_data.index)

filtered_data.drop(['Coordinates', 'Order City', 'Order Country'], axis=1, inplace=True)

# Calculation of the distance between the locations of store and order 
def calculate_distance(row):
    customer_coords = (row['Latitude'], row['Longitude'])
    order_coords = (row['Order Latitude'], row['Order Longitude'])
    return great_circle(customer_coords, order_coords).kilometers

filtered_data['Distance (km)'] = filtered_data.apply(calculate_distance, axis=1)


target_column = 'Late_delivery_risk'
targets = filtered_data.loc[:, target_column]
features = filtered_data.drop(target_column, axis=1)
features['order_date'] = pd.to_datetime(features['order date (DateOrders)'])
features['order_year'] = pd.DatetimeIndex(features['order_date']).year.astype('float64')
features['order_month'] = pd.DatetimeIndex(features['order_date']).month.astype('float64')
features['order_day'] = pd.DatetimeIndex(features['order_date']).day.astype('float64')
features.drop(['order date (DateOrders)', 'order_date'], axis=1, inplace=True)

categorical_features = ['Category Name', 'Customer Zipcode', 'Department Name']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.3)
print(x_train.columns)

# Check for missing data
print('Training set: \n', x_train.isnull().sum())
print('Test set: \n', x_test.isnull().sum())


# numeric_features = x_train.select_dtypes(include=[float, int])
# corr = numeric_features.corr(method='pearson')
# corr.style.background_gradient(cmap='coolwarm')
# fig, ax = plt.subplots()
# sns.heatmap(corr, cmap=plt.get_cmap('coolwarm'), cbar=True, ax=ax)
# plt.savefig('numeric_features.png', bbox_inches='tight', pad_inches=0.0)

# Depending on the cardinality of the categorical features, we are trying to apply different encoding techniques
count_enc_columns = ['Category Name', 'Customer Zipcode', 'Department Name']
count_encoder = CountEncoder(cols=count_enc_columns)
x_train = count_encoder.fit_transform(x_train)
x_test = count_encoder.transform(x_test)

ohe_columns = ['Type', 'Customer Segment', 'Market', 'Shipping Mode', 'Order Status']
x_train = pd.get_dummies(x_train, columns=ohe_columns).reset_index(drop=True)
x_test = pd.get_dummies(x_test, columns=ohe_columns).reset_index(drop=True)

# Standardize the numeric features
scaler = StandardScaler()
preproc_columns = x_train.columns
x_train_std = pd.DataFrame(scaler.fit_transform(x_train), columns=preproc_columns)
x_test_std = pd.DataFrame(scaler.transform(x_test), columns=preproc_columns)

x_train_std = x_train_std.rename(columns={'Latitude': 'Store Latitude',
                                          'Longitude': 'Store Longitude'})
x_test_std = x_test_std.rename(columns={'Latitude': 'Store Latitude',
                                        'Longitude': 'Store Longitude'})

x_train = x_train.rename(columns={'Latitude': 'Store Latitude',
                                  'Longitude': 'Store Longitude'})
x_test = x_test.rename(columns={'Latitude': 'Store Latitude',
                                'Longitude': 'Store Longitude'})

# Save transformed data
joblib.dump(x_train_std, osp.join(processed_data_dir, 'x_train_std.joblib'))
joblib.dump(x_test_std, osp.join(processed_data_dir, 'x_test_std.joblib'))

joblib.dump(x_train, osp.join(processed_data_dir, 'x_train.joblib'))
joblib.dump(x_test, osp.join(processed_data_dir, 'x_test.joblib'))

joblib.dump(y_train, osp.join(processed_data_dir, 'y_train.joblib'))
joblib.dump(y_test, osp.join(processed_data_dir, 'y_test.joblib'))

# Plot correlation matrix
corr = x_train_std.corr(method='spearman')
corr.style.background_gradient(cmap='coolwarm')
fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(corr, cmap=plt.get_cmap('coolwarm'), cbar=True, ax=ax)
plt.savefig('outputs/Spearman_preprocessed_all_features_std.png', bbox_inches='tight', pad_inches=0.0)

import os
import numpy as np
import pandas as pd
from typing import List

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from .count import CountEncoder

from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.metrics import roc_auc_score


def read_data(dir_path, return_X_y=True):
    train_df = pd.read_csv(os.path.join(dir_path, 'train.csv'))

    if return_X_y:
        predictors = [col for col in train_df.columns if col != 'target']
        X = train_df.loc[:, predictors]
        y = train_df.loccategory_encoders[:, 'target']

        return X, y

    return train_df


def bin_feature(df, feature_title, bins=None):
    bins_num = 5
    if isinstance(bins, type(None)):
        bins = pd.cut(df[feature_title], bins=bins_num, retbins=True)[1]

    # For the sake of simplicity, the binned categories have been enumerated
    labels = [f'group{i}' for i in range(1, bins_num+1)]
    binned_feature = pd.cut(df[feature_title], bins=bins, labels=labels, include_lowest=True)

    return binned_feature, bins


def numeric_feature_preprocessor(numeric_features: List[str]):
    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                                ('scaler', RobustScaler())
                                ])

    return ColumnTransformer(transformers=[('numeric', num_transformer, numeric_features)])


def categorical_feature_preprocessor(ohe_columns: List[str],
                                     count_enc_columns: List[str],
                                     ord_enc_columns: List[str],
                                     ordered_categories: List[List[str]]):

    # The passed columns will be one-hot encoded
    oh_encod_preprocessor = ColumnTransformer(transformers=[('oh_encoder', OneHotEncoder(handle_unknown='ignore'),
                                                             pd.Index(ohe_columns))])

    # Here count encoding is applied
    count_encod_preprocessor = ColumnTransformer(transformers=[('count_encoder', CountEncoder(),
                                                                pd.Index(count_enc_columns))])

    # The same transformer for ordinal encoding with the custom class
    transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoderForColumn(ord_enc_columns, ordered_categories))])

    ord_encod_preprocessor = ColumnTransformer(transformers=[('ordinal_encoder', transformer, ord_enc_columns)])

    return [oh_encod_preprocessor, count_encod_preprocessor, ord_encod_preprocessor]


def make_generic_pipeline(num_features_preprocessor: ColumnTransformer,
                          cat_features_preprocessor: List[ColumnTransformer]):
    oh_encod_preprocessor, count_encod_preprocessor, ord_encod_preprocessor = cat_features_preprocessor
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('preprocessor1', num_features_preprocessor),
            ('preprocessor2', oh_encod_preprocessor),
            ('preprocessor3', count_encod_preprocessor),
            ('preprocessor4', ord_encod_preprocessor)
        ]))
    ])
    return pipeline


class OrdinalEncoderForColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column, categories):
        self.column = column
        self.categories = categories
        self.ordinal_encoder = OrdinalEncoder(categories=self.categories,
                                              handle_unknown='error')

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
          X = pd.DataFrame(X, columns=self.column)
        self.ordinal_encoder.fit(X[self.column].values.reshape(-1, 1))
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
          X = pd.DataFrame(X, columns=self.column)
        X_encoded = X.copy()
        X_encoded[self.column] = self.ordinal_encoder.transform(X[self.column].values.reshape(-1, 1))
        return X_encoded

    def get_feature_names_out(self, input_features=None):
        return self.column


def set_features_names(pipeline, df):
    transformed_column_names = []

    for transformer in pipeline.named_steps['features'].transformer_list:
      col_names = transformer[1].get_feature_names_out()
      transformed_column_names.extend([col_name.split('__')[1] for col_name in col_names])

    return pd.DataFrame(df, columns=transformed_column_names)


def remove_correlated_features(df):
    drop_columns = ['Peak_calls_Sum',
                    'OffPeak_calls_Sum',
                    'Weekend_calls_Sum',
                    'National_calls',
                    'Total_call_cost']

    return df.drop(drop_columns, axis=1)


def calc_auc(model, data, targets):
  predictions = model.predict_proba(data)[:, 1]
  print('AUC: ', roc_auc_score(targets, predictions))



import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score,r2_score,mean_absolute_error,mean_squared_error,accuracy_score,classification_report,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor, MLPClassifier
import itertools

def init_models(type='classification'):
    if type == 'regression':
        models = {
            'dt_regressor_pipeline': Pipeline([('scaler', StandardScaler()), ('regressor', DecisionTreeRegressor(random_state=42))]),
            # 'rf_regressor_pipeline': Pipeline([('scaler', StandardScaler()), ('regressor', RandomForestRegressor(random_state=42))]),
            # 'lr_regressor_pipeline': Pipeline([('scaler', StandardScaler()), ('regressor', LinearRegression())]),
            'xgb_regressor_pipeline': Pipeline([('scaler', StandardScaler()), ('regressor', XGBRegressor(random_state=42))]),
            # 'mlp_regressor_pipeline': Pipeline([('scaler', StandardScaler()), ('regressor', MLPRegressor(random_state=42))]),
        }
    else:
        models = {
            'dt_classifier_pipeline': Pipeline([('scaler', StandardScaler()), ('classifier', DecisionTreeClassifier(random_state=42))]),
            # 'rf_classifier_pipeline': Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(random_state=42))]),
            'lr_classifier_pipeline': Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(max_iter=1000, random_state=42))]),
            'xgb_classifier_pipeline': Pipeline([('scaler', StandardScaler()), ('classifier', XGBClassifier(random_state=42))])
        }

    return models

def train_models(models, x_train, y_train):
    for model in models:
        models[model].fit(x_train, y_train)
        print(f"{model} trained")

def predict(models, x):
    preds = {}
    for model in models:
        preds[model] = models[model].predict(x)
        if 'mlp' in model:
            if 'classifier' in model:
                model_part = models[model].named_steps['classifier']
            else:
                model_part = models[model].named_steps['regressor']
            plt.plot(model_part.loss_curve_)
            plt.title('Loss Curve')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.show()
    return preds

def eval_classification(y_true, preds):
    acc_dict = {}
    roc_dict = {}
    rep_dict = {}
    for model in preds:
        acc = accuracy_score(y_true, preds[model])
        roc = roc_auc_score(y_true, preds[model])
        rep = classification_report(y_true, preds[model])
        acc_dict[model] = acc
        roc_dict[model] = roc
        rep_dict[model] = rep
        print(f'Accuracy for {model}: {acc}')
        print(f'ROC for {model}: {roc}')
        print(f'Report for {model}: {rep}')
        ax=plt.subplot()
        sns.heatmap(confusion_matrix(y_true, preds[model]),annot=True,ax=ax)
        ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels')
        ax.set_title('Confusion matrix for Risk Delivery classfication')
    
    return acc_dict, roc_dict, rep_dict

def eval_regression(y_true,preds):
    r2_dict = {}
    mse_dict = {}
    mae_dict = {}
    for model in preds:
        r2 = r2_score(y_true, preds[model])
        mse = mean_squared_error(y_true, preds[model])
        mae = mean_absolute_error(y_true, preds[model])
        r2_dict[model] = r2
        mse_dict[model] = mse
        mae_dict[model] = mae
        print(f'R2 for {model}: {r2}')
        print(f'MSE for {model}: {mse}')
        print(f'MAE for {model}: {mae}')
    return r2_dict, mse_dict, mae_dict

def eval_per_order_status(x, y, models, type):
    order_status_list = ['PENDING', 'PROCESSING', 'PENDING_PAYMENT', 'PAYMENT_REVIEW', 'ON_HOLD', 'PAYMENT_REVIEW', 'COMPLETE', 'CLOSED']
    p1_dict = {}
    p2_dict = {}
    p3_dict = {}
    for model in models:
        for order_status in order_status_list:
            x_order_status = x[x['Order Status_' + order_status] == True]
            y_order_status = y.loc[x_order_status.index]
            print('Order Status:', order_status)
            print('\n')
            print("EVALUATION OF", model)
            pred_order_status = {model: models[model].predict(x_order_status)}
            if type == 'regression':
                p1, p2, p3 = eval_regression(y_order_status["Days for shipping (real)"], pred_order_status)
            else:
                p1, p2, p3 = eval_classification(y_order_status["Late_delivery_risk"], pred_order_status)
            print('\n\n')
            p1_dict[model + '_' + order_status] = p1
            p2_dict[model + '_' + order_status] = p2
            p3_dict[model + '_' + order_status] = p3
    return p1_dict, p2_dict, p3_dict

def get_ft_importance(x, model, type):
    if type == 'regression':
        model_part = model.named_steps['regressor']
    else:
        model_part = model.named_steps['classifier']
    importance = model_part.feature_importances_
    feature_names = x.columns if hasattr(x, 'columns') else [f'feature_{i}' for i in range(x.shape[1])]
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    return importance_df

def get_best_fts(prep_data, models, features, features_to_drop, type):
    performance_dict = {}
    for model in models:
        performance_dict[model] = {}
        for r in range(1, len(features) + 1):
            for combination in itertools.combinations(features, r):
                print(f'Model: {model}')
                print(f"Combination: {combination}")
                non_selected_features = list(combination) + features_to_drop
                combined_pattern = '|'.join(non_selected_features)
                simple_prep_data = prep_data.drop(prep_data.filter(regex=combined_pattern).columns, axis=1)
                x = simple_prep_data.drop(['Late_delivery_risk', 'Days for shipping (real)'],axis=1)
                if type == 'regression':
                    y = simple_prep_data[['Days for shipping (real)']]
                else:
                    y = simple_prep_data[['Late_delivery_risk']]
                # train-test_split
                x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.20,random_state=42,stratify=y)
                # train-validation split
                x_train,x_val,y_train,y_val = model_selection.train_test_split(x_train,y_train,test_size=0.25,random_state=42,stratify=y_train)

                models[model].fit(x_train, y_train)
                if type == 'regression':
                    performance_dict[model][combination] = mean_absolute_error(y_val["Days for shipping (real)"], models[model].predict(x_val))
                else:
                    performance_dict[model][combination] = roc_auc_score(y_val["Late_delivery_risk"], models[model].predict(x_val))
                print(f"Performance: {performance_dict[model][combination]}")
                print('\n')
        
        performance_dict[model] = dict(sorted(performance_dict[model].items(), key=lambda item: item[1], reverse=True))
        print(f"Best features for {model}: {list(performance_dict[model].keys())[0]}")
    return performance_dict

def init_nn(type='regression'):
    if type == 'regression':
        models = {
            'mlp_regressor_pipeline': Pipeline([('scaler', StandardScaler()), 
                                                ('regressor', MLPRegressor(
                                                    hidden_layer_sizes=(100, 50, 25),
                                                    random_state=42,
                                                    batch_size=64,
                                                    shuffle=True,
                                                    early_stopping=True,
                                                    tol=1e-4,
                                                    n_iter_no_change=50,
                                                    learning_rate_init=0.0001,
                                                    activation='relu',
                                                    ))]),
        }
    else:
        models = {
            'mlp_classifier_pipeline': Pipeline([('scaler', StandardScaler()), 
                                                 ('classifier', MLPClassifier(
                                                    hidden_layer_sizes=(100, 50, 25),
                                                    random_state=42,
                                                    batch_size=64,
                                                    shuffle=True,
                                                    early_stopping=True,
                                                    tol=1e-4,
                                                    n_iter_no_change=50,
                                                    learning_rate_init=0.0001,
                                                    activation='relu',
                                                    # max_iter=10
                                                    ))])
        }
    return models
        
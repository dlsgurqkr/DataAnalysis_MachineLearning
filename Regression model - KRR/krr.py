import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def wave(x: np.ndarray) -> np.ndarray:
    return np.exp(-(x)**2)


def get_train_test_split(filename: str) -> dict:
    
    # Split the data into training and testing sets with 20% of the data as test
    x_tv, x_test, y_tv, y_test = train_test_split(df.x , df.y, shuffle=True, test_size=0.2)
    
    # Construct the dictionary to return
    split_data = {
        "train": {
            "x": x_tv.values.reshape(-1,1),
            "y": y_tv.values
        },
        "test": {
            "x": x_test.values.reshape(-1,1),
            "y": y_test.values
        }
    }
    
    return split_data


def get_hyper_model() -> GridSearchCV:
    
    parameters = {
        'alpha':[0.1,1,1],
        'kernel':['linear','rbf','poly'],
        'gamma':[0.01,0.1,1]
    }
    
    model = KernelRidge()
    
    hypermodel = GridSearchCV(model,parameters,cv=8,scoring='neg_mean_squared_error')

    
    return hypermodel


def train(hyper_model: GridSearchCV, data: dict,name:str) -> KernelRidge:
    
    dt = get_train_test_split(data)
    
    x_tv = dt['train']['x']
    y_tv = dt['train']['y']
    
    x_test = dt['test']['x']
    y_test = dt['test']['y']
    
    #fit model using training data
    hyper_model.fit(x_tv, y_tv)
    
    #Find the best esitmate
    hyper_model.best_estimator_
    
    y_pred_train = hyper_model.predict(x_tv.reshape(-1,1))
    y_pred_test = hyper_model.predict(x_test.reshape(-1,1))
    
    
    mae_train = mean_absolute_error(y_tv, y_pred_train)
    mse_train = mean_squared_error(y_tv, y_pred_train)
    
    r2_train = r2_score(x_tv, y_tv)
    
    mae_test = mean_absolute_error(y_test,y_pred_test)
    mse_test = mean_squared_error(y_test,y_pred_test)
    
    r2_test  = r2_score(x_test,y_test)
    

    result = {
        "best_params": {
            "alpha": hyper_model.best_estimator_.alpha,
            "gamma": hyper_model.best_estimator_.gamma,
            "kernel": "rbf"
            
        },
        
        "train": {
            "mae": mae_train,
            "mse": mse_train,
            "r2":  r2_train
        },
        
        "test": {
            "mae": mae_test,
            "mse": mse_test,
            "r2":  r2_test
        }
    }
    
    with open(f"{name}.json",'w') as file:
        json.dump(result, file, indent=2)
        
    with open(f"{name}.pickl",'wb') as file:
        pickle.dump(hyper_model.best_estimator_, file )
    
    return result, hyper_model.best_estimator_



def plot(model: KernelRidge, data: dict, name: str) -> mpl.figure.Figure:
    fig = plt.figure()
    return fig


if __name__ == "__main__":
    data = get_train_test_split("wave.csv")

    hyper_model = get_hyper_model()

    best_model = train(hyper_model, data, "ex13")

    fig = plot(best_model, data, "ex13")

    plt.show()

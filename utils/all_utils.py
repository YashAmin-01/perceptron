import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os
from matplotlib.colors import ListedColormap

plt.style.use('fivethirtyeight')

def prepare_data(df):
    """used to separate depndent and independent features 

    Args:
        df (pd.DataFrame): pandas Dataframe

    Returns:
        tuple: returns tuples of dependent and independent variables
    """
    x = df.drop('y', axis=1)
    y = df['y']
    return x,y

def save_model(model, filename):
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    filePath = os.path.join(model_dir, filename)
    joblib.dump(model, filePath)

def save_plot(df, file_name, model):
    def _create_base_plot(df):
        df.plot(kind='scatter', x='x1', y='x2', c='y', s=100, cmap='winter')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        figure = plt.gcf()
        figure.set_size_inches(10,8)
    
    def _plot_decision_region(x,y,classfier,resolution=.02):
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[: len(np.unique(y))])
        
        x = x.values
        x1 = x[:, 0]
        x2 = x[:, 1]
        x1_min, x1_max = x1.min() - 1, x1.max() + 1
        x2_min, x2_max = x2.min() - 1, x2.max() + 1
        
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        print(xx1)
        print(xx1.ravel())
        
        z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        z = z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, z, alpha=0.2, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        
        plt.plot()
    
    x,y = prepare_data(df)
    
    _create_base_plot(df)
    _plot_decision_region(x,y,model)
    
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    plotPath = os.path.join(plot_dir, file_name)
    plt.savefig(plotPath)
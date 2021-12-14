from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np

def main(data, eta, epochs, filename, plotFileName):
    
    df = pd.DataFrame(data)

    print(df)

    x,y = prepare_data(df)

    eta = 0.3 #between 0 and 1
    epochs = 10

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(x,y)

    _ = model.total_loss()

    save_model(model, filename=filename)

    save_plot(df, plotFileName, model)

if __name__ == '__main__':
    OR = {
        'x1': [0,0,1,1],
        'x2': [0,1,0,1],
        'y': [0,1,1,1]
    }

    eta = 0.3
    epochs = 10

    main(data = OR, eta = eta, epochs = epochs, filename = 'or.model', plotFileName = 'or.png')
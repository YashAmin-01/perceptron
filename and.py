from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np

AND = {
    'x1': [0,0,1,1],
    'x2': [0,1,0,1],
    'y': [0,0,0,1]
}

df = pd.DataFrame(AND)

print(df)

x,y = prepare_data(df)

eta = 0.3 #between 0 and 1
epochs = 10

model = Perceptron(eta=eta, epochs=epochs)
model.fit(x,y)

_ = model.total_loss()

save_model(model, filename='and.model')

save_plot(df, 'and.png', model)
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")

def main(data, eta, epochs, filename, plotFileName):
    
    df = pd.DataFrame(data)

    logging.info(f"This is the actual DataFrame {df}")

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

    try:
        logging.info('>>>>>starting training >>>>>')
        main(data = OR, eta = eta, epochs = epochs, filename = 'or.model', plotFileName = 'or.png')
        logging.info('<<<<<training done successfully <<<<<')
    except Exception as e:
        logging.exception(e)
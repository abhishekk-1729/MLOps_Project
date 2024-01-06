import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):

    @abstractmethod
    def train(self, X_train, y_train):
        pass

class LinearRegressionModel(Model):

    try:
        def train(self, X_train, y_train):
            reg = LinearRegression()
            reg.fit(X_train,y_train)
            logging.info("Model training completed")
            return reg
    
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e




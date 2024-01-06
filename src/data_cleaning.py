import logging
from abc import ABC, abstractmethod

from pandas.core.api import Series as Series
"""
ABC: abstract base classes.
Python does not support abstact classes internally so this is the library to be used
We can define abtract methods: methods which are compulsory in every sub class[inherited classes]
"""
from typing import Union
"""
var: Union(int, str) = "abc"
"""
import numpy as np
import pandas as pd

"""
Pandas: Series and dataframe: like key value pairs where normal indexing is 0,1,2 but can be changed according to the requirement
Series: single column
Dataframe: multiple columns

"""

from sklearn.model_selection import train_test_split
""" 
train_test_split splits your data(X -> Y) and return X_train, X_test, y_train, y_test'
Suppose there are 20 entries. What we can do is divide the data accordingly into X_train and X_test 
in some ration(8:2) so that some data can be used for training and some for checking the accuracy of the model

"""

class DataStrategy(ABC):

    @abstractmethod
    def handle_data(self,data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        try:
            # data = data.drop(
            #     [
            #         "order_approved_at",
            #         "order_delivered_carrier_date",
            #         "order_delivered_customer_date",
            #         "order_estimated_delivery_date",
            #         "order_purchase_timestamp",
            #     ],
            #     axis=1
            # )

            """
            Data.drop, drop certain columns or indexes, axis means drop the columns not the indexes
            Data.fillna: null values cells for a particular column can be filled according to the requirement
            inplace=true: permanently change
            """

            data["product_weight_g"].fillna(data["product_weight_g"].median(),inplace=True)
            data["product_length_cm"].fillna(data["product_weight_g"].median(),inplace=True)
            data["product_height_cm"].fillna(data["product_weight_g"].median(),inplace=True)
            data["product_width_cm"].fillna(data["product_weight_g"].median(),inplace=True)
            # data["review_comment_message"].fillna("No Review",inplace=True)

            data = data.select_dtypes(include=[np.number]) # To have columns with numerical value only
            # cols_to_drop  = ["customer_zip_code_prefix","order_item_id"]
            # data = data.drop(cols_to_drop,axis=1)

            return data

        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e

class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        try:
            X = data.drop(["review_score"],axis=1)
            y = data["review_score"]
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
            # in case of random_state you will get same data 
            return X_train,X_test,y_train,y_test
        except Exception as e:
            logging.error("Error in dividing data : {}".format(e))
            raise e
        

class DataCleaning:
    #preprocess and divide
    def __init__(self,data:pd.DataFrame,strategy:DataStrategy):
        self.data=data
        self.strategy=strategy

    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e
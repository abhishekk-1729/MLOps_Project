import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning,DataPreProcessStrategy,DataDivideStrategy
from typing import Tuple
from typing_extensions import Annotated

"""
Tuple means what all to be returned, different from union as union don't give a proper structure what to return.
Annotated: On its own Annotated does not do anything other than assigning extra information (metadata) to a reference.
"""

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"X_train"], 
    Annotated[pd.DataFrame,"X_test"], 
    Annotated[pd.Series,"y_train"], 
    Annotated[pd.Series,"y_test"]  
]:
    
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df,process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data,divide_strategy)
        X_train,X_test,y_train,y_test=data_cleaning.handle_data()
        return  X_train,X_test,y_train,y_test

    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e





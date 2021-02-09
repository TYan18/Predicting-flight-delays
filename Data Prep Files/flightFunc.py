import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def txt_to_csv(filename, scaler, output_name):
    # filename (str): the name of the .txt file, including "".txt"
    # extension and local folder path
    
    # scaler (method): either StandardScaler() or MinMaxScaler() from
    # sklearn.processing
    
    # output_name (str): the name to append to "x_<output_name>.csv" and
    # "y_<output_name>.csv"
    
    # Import the .txt file (parameter) as a Pandas df
    dfRead = pd.read_csv(filename, delimiter = '\t')
    
    # Drop rows where target value is an outlier (arrival more than
    # 42.5 minutes late or more than 49.5 minutes early)
    dfRead = dfRead.loc[(dfRead['arr_delay'] <= 42.5) &
                        (dfRead['arr_delay'] >= -49.5)]
    
    # Separate the numerical feature columns for scaling
    dfNumeric = dfRead[['crs_dep_time', 'crs_arr_time',
                              'crs_elapsed_time', 'distance']]
    
    # Convert the time column into hours
    dfNumeric.loc[:, 'crs_arr_time'] = dfRead.loc[:, 'crs_arr_time'] / 100
    
    # Scale the numerical data either with StandardScaler() or
    # MinMaxScaler() from parameter passed in function
    dfNumeric_scaled = scaler.fit_transform(dfNumeric)
    
    # Convert the scaled X Numpy array back into Pandas df with the 
    # original column names and "FT" (fit-transformed)
    colList = dfNumeric.columns
    colList = [col + 'FT' for col in colList]
    dfNumeric_scaled = pd.DataFrame(dfNumeric_scaled, columns = colList)
    
    # Convert numerical ID data into objects for one-hot encoding
    df = dfRead.astype({
    'mkt_carrier_fl_num': object,
    'op_carrier_fl_num': object,
    'origin_airport_id': object,
    'dest_airport_id': object
    })
    
    # Select all of the object-type columns for one-hot encoding
    dfObjects = df.select_dtypes(include = 'object')
    
    # Convert the flight date column to a datetime format, extract the
    # month, and then delete this column
    dfObjects['month'] = pd.DatetimeIndex(pd.to_datetime(df['fl_date'],
                        infer_datetime_format = True)).month
    dfObjects = dfObjects.drop('fl_date', axis = 1, errors = 'ignore')
    
    # Concatenate the object-type columns with features dataframe
    dfNumeric_scaled = dfNumeric_scaled.reset_index(drop = True)
    dfObjects = dfObjects.reset_index(drop = True)
    dfConcat = pd.concat([dfNumeric_scaled, dfObjects], axis = 1)
    
    #Drop columns with features highly correlated to features in other
    # columns
    dfConcat.drop(columns = ['mkt_unique_carrier', 'branded_code_share',
                           'mkt_carrier', 'mkt_carrier_fl_num',
                           'origin_airport_id', 'origin_city_name',
                           'dest_airport_id', 'dest_city_name', 'dup',
                            'tail_num', 'op_carrier_fl_num', 'arr_delay',
                            'op_carrier_fl_num'],
                          inplace = True, errors = 'ignore')
    
    # Preserve the sign of the target variable (+ for delay, and - for
    # early arrival), then scale the values
    y_sign = ((dfRead['arr_delay'] > 0) * 1)
    y = StandardScaler().fit_transform(dfRead['arr_delay'].
                                       values.reshape(-1, 1))
    
    # Use one-hot encoding to create dummy variables for categorical
    # features, designate the correct columns for the target values as well
    # as preserved sign, and drop all rows with NaN values
    FinalDF = pd.get_dummies(dfConcat)
    FinalDF['yFT'] = y
    FinalDF['y_sign'] = y_sign
    FinalDF.dropna(inplace = True)

    # Write the feature columns and target columns to "X" and "y" .csv files
    # (parameter "output_name" is appended)
    FinalDF.iloc[:,:-2].to_csv('X_' + output_name + '.csv', index = False)
    FinalDF.iloc[:,-2:].to_csv('y_' + output_name + '.csv', index = False)
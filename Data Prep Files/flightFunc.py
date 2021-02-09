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
    dfNumeric.loc[:, 'crs_dep_time'] = dfRead.loc[:, 'crs_dep_time'] // 100
    dfNumeric.loc[:, 'crs_arr_time'] = dfRead.loc[:, 'crs_arr_time'] // 100
    
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
    y = scaler.fit_transform(dfRead['arr_delay'].
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
    FinalDF.iloc[:,:-2].to_csv('X_' + output_name + '.csv', index = False, compression = 'gzip')
    FinalDF.iloc[:,-2:].to_csv('y_' + output_name + '.csv', index = False, compression = 'gzip')
    
def replaceCarrierMonthWithAvgDelay(df):
    carrierDict={
        '9E': 3.788253768330484,
        '9K': -1.4138972809667674,
        'AA': 6.209127910387774,
        'AS': 0.4580709294158264,
        'AX': 15.614108372836077,
        'B6': 11.328905876893792,
        'C5': 23.297226405497323,
        'CP': 5.752920400632577,
        'DL': 0.4649172663471822,
        'EM': 6.439237738206811,
        'EV': 11.460218091834946,
        'F9': 11.294148868243784,
        'G4': 8.948751471435369,
        'G7': 8.645223084384094,
        'HA': 0.7479588660633051,
        'KS': 17.362094951017333,
        'MQ': 6.192537786526977,
        'NK': 5.135043464348568,
        'OH': 7.457289195029419,
        'OO': 7.155529056303303,
        'PT': 4.442924183761644,
        'QX': 2.056287279578388,
        'UA': 7.0310394918378565,
        'VX': 1.7279776132454965,
        'WN': 3.549975537488939,
        'YV': 9.682003383338802,
        'YX': 3.946617621243796,
        'ZW': 7.347722443129507
    }
    
    df['carrierAvgDelay']=df['op_unique_carrier'].apply(lambda y: carrierDict[y])
    df.drop(columns='op_unique_carrier', inplace=True)
    
    monthDict={
        1: 3.9587876597858975,
        2: 6.745095564322296,
        3: 2.818772851018936,
        4: 4.159130781653622,
        5: 6.511144454809372,
        6: 10.414443676520804,
        7: 8.977515143605581,
        8: 8.898890118485891,
        9: 1.70845239337149,
        10: 2.8535847853724894,
        11: 2.9936067372926765,
        12: 5.110635893078993
    }
    
    df['MonthAvgDelay']=df['month'].apply(lambda y: monthDict[y])
    df.drop(columns='month', inplace=True)
    
    return df

def replace_origin_dest(df):
    # df: X features dataframe without one-hot encoding

    # Find the average delay times by origin location, and store the values in a dictionary
    origin = pd.read_csv('origin_arr_delay.txt', delimiter = '\t', names = ['origin', 'avg_delay'])
    origin = pd.Series(origin.avg_delay.values, index = origin.origin).to_dict()
    
    # Find the average delay times by destination location, and store the values in a dictionary
    dest = pd.read_csv('dest_arr_delay.txt', delimiter = '\t', names = ['dest', 'avg_delay'])
    dest = pd.Series(dest.avg_delay.values, index = dest.dest).to_dict()
    
    # Replace the values in the "origin" and "dest" columns with the average arrival delay time
    df['origin'] = df['origin'].replace(origin)
    df['dest'] = df['dest'].replace(dest)
    
    return df
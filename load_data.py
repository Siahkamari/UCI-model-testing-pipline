import pandas as pd
import urllib.request
import numpy as np
from IPython.display import display
from zipfile import ZipFile

def cut_data(y, X, cutoff):
  X_train = X[:cutoff]
  y_train = y[:cutoff]
  X_test = X[cutoff:]
  y_test = y[cutoff:]
  return y_train, X_train, y_test, X_test

def add_lag(df, targets, lag_terms=[1,2,3]):
  lag_var = []
  for lag in lag_terms:
    for target in targets:
      lag_var.append(f'{target}_lag_{lag}')
      df[lag_var[-1]] = df[target].shift(lag)
  return df, lag_var

def one_hot(df, categorical_col_names):
  for name in categorical_col_names:
      one_hot = pd.get_dummies(df[name])
      df = df.drop(name, axis = 1)
      df = pd.concat([df, one_hot], axis=1)
  return df

def create_ls_lag_1(df, targets):
    diff_terms = []
    for target in targets:
        diff_terms.append(f'{target}_ls_lag_1')
        df[diff_terms[-1]] = df[target] - df[f'{target}_lag_1']
        df = df.drop(target, axis = 1)
    return df, diff_terms
    
class load_data():
  def air_quality(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip'
    urllib.request.urlretrieve(url, 'data/regression/air_quality.zip')

    with ZipFile('data/regression/air_quality.zip') as myzip:
        with myzip.open('AirQualityUCI.csv') as myfile:
          df = pd.read_csv(myfile ,delimiter=';',decimal=',')

    df.drop(df.columns[[15,16]],axis = 1, inplace=True) # these extra columns are created due to bad format
    df.drop(df.index[9357:],axis = 0, inplace=True) # rows at the end are empty

    # dealing with the missing values
    df.replace({-200.0: None}, inplace=True) # -200.0 stands for None as (see data description)
    # display(df.isnull().sum())
    df.drop('NMHC(GT)', axis=1, inplace=True) # %90 of values are None so lets get rid of this target
    df.fillna(method = 'ffill', inplace=True)

    # converting time to hour in float format
    hour = pd.to_datetime(df.iloc[:,1],format='%H.%M.%S') - pd.to_datetime('00.00.00',format='%H.%M.%S')
    hour /= pd.to_timedelta(1, unit='H')
    df['Time']= hour
    df['Date'] = pd.to_datetime(df.Date)
    df.set_index('Date', inplace=True)

    targets = ['CO(GT)','C6H6(GT)','NOx(GT)','NO2(GT)']
    df, lag_var = add_lag(df, targets)
    df.fillna(method = 'bfill', inplace=True)
    display(df.head())

    features = ['Time','PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH']
    features += lag_var

    X = df[features].to_numpy(dtype=np.float32)
    y = df[targets].to_numpy(dtype=np.float32)

    # We will use data points in 2004 as train and 2005 as test
    return cut_data(y, X, 7110)
      
  def airfoil_self_noise(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat'
    urllib.request.urlretrieve(url, 'data/regression/airfoil_self_noise.csv')

    col_names = ['Frequency','Angle of attack','Chord length','Velocity','Thickness','Sound pressure']
    df = pd.read_csv('data/regression/airfoil_self_noise.csv',delimiter='\t',names=col_names)
    display(df.head())

    X = df.drop("Sound pressure",axis = 1).to_numpy(dtype=np.float32)
    y = df["Sound pressure"].to_numpy(dtype=np.float32)
    return y, X

  def bias_correction_ucl(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00514/Bias_correction_ucl.csv'
    urllib.request.urlretrieve(url, 'data/regression/Bias_correction_ucl.csv')

    df = pd.read_csv('data/regression/Bias_correction_ucl.csv' ,delimiter=',')
    df['Date'] = pd.to_datetime(df.Date)
    df.set_index('Date', inplace = True)
    df.fillna(method = 'ffill', inplace=True)

    categorical_col_names = ['station']
    df = one_hot(df, categorical_col_names)

    df['bias_min'] = df['LDAPS_Tmin_lapse'] - df['Next_Tmin']
    df['bias_max'] = df['LDAPS_Tmax_lapse'] - df['Next_Tmax']

    targets = ['bias_min','bias_max']
    df, lag_var = add_lag(df, targets)
    df.fillna(method = 'bfill', inplace=True)

    display(df.head())

    X = df.drop(['Next_Tmin','Next_Tmax','bias_min','bias_max'],axis=1).to_numpy(dtype=np.float32)

    # We take the bias in temperature prdicted by the LDAPS model as target

    y = df[['bias_min','bias_max']].to_numpy(dtype=np.float32)

    # We will use data points in summer 2017 as test and summer  2013-2016 as training
    return cut_data(y, X, cutoff=6200)
  
  def CCPP(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip'
    urllib.request.urlretrieve(url, 'data/regression/CCPP.zip')

    with ZipFile('data/regression/CCPP.zip') as myzip:
      with myzip.open('CCPP/Folds5x2_pp.xlsx') as myfile:
        df = pd.read_excel(myfile)
    display(df.head())

    X = df.drop(['PE'],axis=1).to_numpy(dtype=np.float32)
    y = df['PE'].to_numpy(dtype=np.float32)
    return y, X
  
  def communities(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data'
    urllib.request.urlretrieve(url, 'data/regression/communities.csv')
    df = pd.read_csv('data/regression/communities.csv',header=None)

    display(df.head())
    # Lets get rid of first 5 the nonpredictve features (country, state, ...)
    df.drop(df.columns[0:5],axis=1, inplace=True)

    # Lets replace missing values with mean of the columns
    df.replace({'?': None}, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(df.mean(), inplace= True)

    X = df.iloc[:,:-1].to_numpy(dtype=np.float32)
    y = df.iloc[:,-1].to_numpy(dtype=np.float32)
    return y, X
  
  def concrete_data(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
    urllib.request.urlretrieve(url, 'data/regression/Concrete_Data.csv')

    df = pd.read_excel('data/regression/Concrete_Data.csv')
    display(df.head())

    X = df.iloc[:,:-1].to_numpy(dtype=np.float32)
    y = df.iloc[:,-1].to_numpy(dtype=np.float32)
    return y, X

  def geographical_original_of_music(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00315/Geographical%20Original%20of%20Music.zip'
    urllib.request.urlretrieve(url, 'data/regression/Geographical_Original_of_Music.zip')

    with ZipFile('data/regression/Geographical_Original_of_Music.zip') as myzip:
      with myzip.open('Geographical Original of Music/default_features_1059_tracks.txt') as myfile:
        df = pd.read_csv(myfile,header=None)
    display(df.head())

    X = df.iloc[:,:-2].to_numpy(dtype=np.float32)
    y = df.iloc[:,-2:].to_numpy(dtype=np.float32)
    return y, X
  
  def parkinson_updrs(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data'
    urllib.request.urlretrieve(url, 'data/regression/parkinsons_updrs.csv')

    df = pd.read_csv('data/regression/parkinsons_updrs.csv')
    #     one_hot = pd.get_dummies(df['subject#'])
    #     df = pd.concat([df, one_hot], axis=1)

    df['test_time2'] = df['test_time'].copy()
    df.set_index('test_time', inplace=True)

    targets = ['motor_UPDRS','total_UPDRS']
    lag_terms = [1,2,3]
    lag_var = []
    for target in targets:
        for lag in lag_terms:
            lag_var.append(f'{target}_lag_{lag}')
            df[lag_var[-1]] = \
            df.groupby('subject#')[target].shift(lag)

    df.fillna(0, inplace=True)

    test_time_3rd_quantile = df['test_time2'].quantile(0.75)
    df_train = df[df['test_time2'] <= test_time_3rd_quantile]
    df_test = df[df['test_time2'] > test_time_3rd_quantile ]
    df_train = df_train.drop(columns =['test_time2'] )
    df_test = df_test.drop(columns =['test_time2'] )
    
    display(df_train.head())

    X_train = df_train.drop(columns = targets).to_numpy(dtype=np.float32)
    y_train = df_train[targets].to_numpy(dtype=np.float32)
    X_test = df_test.drop(columns = targets).to_numpy(dtype=np.float32)
    y_test = df_test[targets].to_numpy(dtype=np.float32)
    return y_train, X_train, y_test, X_test

  def solar_flare(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/solar-flare/flare.data2'
    urllib.request.urlretrieve(url, 'data/regression/solar_flare.csv')
    df = pd.read_csv('data/regression/solar_flare.csv',delimiter=' ', header=None,   skiprows=[0])
    display(df.head())
    y = df.iloc[:,10:12].to_numpy(dtype=np.float32)
    df.drop(df.columns[[10,11]],axis = 1, inplace=True)
    for i in range(3):
      one_hot = pd.get_dummies(df.iloc[:,i])
      df = df.join(one_hot, rsuffix=str(i))
    X = df.iloc[:,3:].to_numpy(dtype=np.float32)
    return y, X
  
  def wine_quality(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    urllib.request.urlretrieve(url, 'data/regression/winequality-white.csv')
    df = pd.read_csv('data/regression/winequality-white.csv',delimiter=';')
    display(df.head())
    X = df.iloc[:,0:11].to_numpy(dtype=np.float32)
    y = df.iloc[:,11].to_numpy(dtype=np.float32)
    return y, X

  def sml2010(self):
    ''' There are two text files, we use both. First two columns are data and time.'''
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00274/NEW-DATA.zip'
    urllib.request.urlretrieve(url, 'data/regression/sml2010.zip')

    with ZipFile('data/regression/sml2010.zip') as myzip:
        # read first data file
        with myzip.open('NEW-DATA-1.T15.txt') as myfile:
            df1 = pd.read_csv(myfile ,delimiter=' ', comment='#', header=None)
        # read column headers
        with myzip.open('NEW-DATA-1.T15.txt') as myfile:
            header = myfile.readlines()[0]
        # read second data file
        with myzip.open('NEW-DATA-2.T15.txt') as myfile:
            df2 = pd.read_csv(myfile ,delimiter=' ', comment='#', header=None)

    df = pd.concat([df1,df2])
    df.columns = header.decode('utf-8').strip('# \n').split()

    df.set_index('1:Date', inplace=True)

    # converting time to hour in float format
    hour = pd.to_datetime(df.iloc[:,0],format='%H:%M') - pd.to_datetime('00:00',format='%H:%M')
    hour /= pd.to_timedelta(1, unit='H')
    df['2:Time']= hour

    # Predict dining room temperature and bedroom temperature
    targets = ['3:Temperature_Comedor_Sensor','4:Temperature_Habitacion_Sensor']
    df, lag_var =  add_lag(df, targets)
    df, targets = create_ls_lag_1(df, targets)
    df.fillna(0, inplace=True)

    display(df.head())

    features = ['2:Time',
                 '5:Weather_Temperature',
                 '6:CO2_Comedor_Sensor',
                 '7:CO2_Habitacion_Sensor',
                 '8:Humedad_Comedor_Sensor',
                 '9:Humedad_Habitacion_Sensor',
                 '10:Lighting_Comedor_Sensor',
                 '11:Lighting_Habitacion_Sensor',
                 '12:Precipitacion',
                 '13:Meteo_Exterior_Crepusculo',
                 '14:Meteo_Exterior_Viento',
                 '15:Meteo_Exterior_Sol_Oest',
                 '16:Meteo_Exterior_Sol_Est',
                 '17:Meteo_Exterior_Sol_Sud',
                 '18:Meteo_Exterior_Piranometro',
                 #'19:Exterior_Entalpic_1', # All 0
                 #'20:Exterior_Entalpic_2', # All 0
                 #'21:Exterior_Entalpic_turbo', # All 0
                 '22:Temperature_Exterior_Sensor',
                 '23:Humedad_Exterior_Sensor',
                 '24:Day_Of_Week']
    features += lag_var 

    X = df[features].to_numpy(dtype=np.float32)
    y = df[targets].to_numpy(dtype=np.float32)

    return cut_data(y[:,-1], X, cutoff=3000)
        
  def bike_sharing(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv'
    urllib.request.urlretrieve(url, 'data/regression/bike_sharing.csv')

    df = pd.read_csv('data/regression/bike_sharing.csv',encoding = "ISO-8859-1")
    df.set_index('Date', inplace=True)
    
    categorical_col_names = ['Seasons','Holiday','Functioning Day']
    df = one_hot(df, categorical_col_names)
    targets = ['Rented Bike Count']
    df, lag_var = add_lag(df, targets)
    df.fillna(method = 'bfill', inplace=True)

    
    # Exclude 'Autumn' because only non-zero at end of dataset
    df = df.drop('Autumn', axis=1)
    display(df.head())
    
    X = df.loc[:, (df.columns != 'Rented Bike Count' ) & (df.columns != 'Date')]\
                  .to_numpy(dtype=np.float32)
    y = df['Rented Bike Count'].to_numpy(dtype=np.float32) # Hourly rented bike count
    
    return cut_data(y, X, 6570)

  def garment_productivity(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00597/garments_worker_productivity.csv'
    urllib.request.urlretrieve(url, 'data/regression/garments_worker_productivity.csv')

    df = pd.read_csv('data/regression/garments_worker_productivity.csv')
    
    # Replace blanks in wip (work in progress) with 0
    df['wip'] = df['wip'].fillna(0)
    
    df['date'] = pd.to_datetime(df.date)
    df.set_index('date', inplace=True)

    # clean 'department' column (remove extraneous white space)
    def string_strip(x):
        return x.strip()
    df['department'] = df['department'].apply(string_strip)
    
    df.drop(columns='day', inplace=True)

    categorical_col_names = ['quarter','department','team']
    targets = ['actual_productivity']

    lag_var = []
    lag_terms = [1,2,3]
    for lag in lag_terms:
      for target in targets:
        for category in categorical_col_names:
            lag_var.append(f'{category}_{target}_lag_{lag}')
            df[lag_var[-1]] = df.groupby(category)[target].shift(lag).fillna(method = 'bfill', inplace=True)

    df.fillna(0, inplace=True)
    df = one_hot(df, categorical_col_names) 

    df_train = df[ (df['Quarter1'] | df['Quarter2'] | df['Quarter3' ])== 1 ]
    df_test = df[(df['Quarter4'] | df['Quarter5'])==1]
    
    display(df_train.head())

    X_train = df_train.loc[:, df_train.columns != 'actual_productivity'].to_numpy(dtype=np.float32)
    y_train = df_train['actual_productivity'].to_numpy(dtype=np.float32)
    X_test = df_test.loc[:, df_test.columns != 'actual_productivity'].to_numpy(dtype=np.float32)
    y_test = df_test['actual_productivity'].to_numpy(dtype=np.float32)
    # predict actual_productivity based on other columns
    return y_train, X_train, y_test, X_test

  ## everything bellow here is classification
  
  def adult(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    urllib.request.urlretrieve(url, 'data/regression/adult-data.csv')
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    urllib.request.urlretrieve(url, 'data/regression/adult-data-test.csv')

    col_names = ['age', 'workclass','fnlwgt','education','education-num','marital-status','occupation',
             'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']

    df = pd.read_csv('data/regression/adult-data.csv', header=None, names = col_names)
    df_test = pd.read_csv('data/regression/adult-data-test.csv', header=None, names = col_names)
    df_test = df_test.drop(df_test.index[0])

    display(df.head())

    df = df.replace({' ?': None})
    df = df.fillna(method = 'backfill')

    df_test = df_test.replace({' ?': None})
    df_test = df_test.fillna(method = 'backfill')

    y_train = df.salary == ' >50K'
    y_test = df_test.salary == ' >50K.'

    y_train = y_train.to_numpy(dtype=np.float32)
    y_test = y_test.to_numpy(dtype=np.int32)

    df = df.drop('salary',axis = 1)
    df_test = df_test.drop('salary',axis = 1)

    col_names = ['workclass','education','marital-status','occupation',
            'relationship','race','sex']

    df = one_hot(df, col_names)  
    df_test = one_hot(df_test, col_names)
        
    df = df.drop('native-country',axis = 1)
    df_test = df_test.drop('native-country',axis = 1)
    df_test.iloc[:,0] = pd.to_numeric(df_test.iloc[:,0] , errors='coerce')

    X_train = df.to_numpy(dtype=np.float32)
    X_test = df_test.to_numpy(dtype=np.float32)
    return y_train, X_train, y_test, X_test

  def iris(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    urllib.request.urlretrieve(url, 'data/classification/iris.csv')

     col_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    df = pd.read_csv('data/classification/iris.csv',  names=col_names)
    display(df.head())

    X = df.drop(columns=['Iris-setosa']).to_numpy(dtype=np.float32)
    y = 2*(df['Iris-setosa'] == 'Iris-virginica') + (df['Iris-setosa'] == 'Iris-versicolor') + 1
    y = y.to_numpy(dtype=np.int32)
    return y, X

  def balance_scale(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data'
    urllib.request.urlretrieve(url, 'data/classification/balance-scale.csv')

    names=['Class Name', 'Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance']
    df  = pd.read_csv('data/classification/balance-scale.csv', header=None, names = names)
    display(df.head())

    X = df.drop(columns='Class Name').to_numpy(dtype=np.float32)
    y = ((df['Class Name'] =='B') + 2*(df['Class Name'] =='R') + 1).to_numpy(dtype=np.int32)
    return y, X

  def transfusion(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data'
    urllib.request.urlretrieve(url, 'data/classification/transfusion.csv')

    df = pd.read_csv('data/classification/transfusion.csv' ,delimiter=',')
    display(df.head())

    X = df.drop(columns=df.columns[-1]).to_numpy(dtype=np.float32)
    y = ((df[df.columns[-1]] ==1) + 1).to_numpy(dtype=np.int32)
    return y, X
  
  def ionosphere(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
    urllib.request.urlretrieve(url, 'data/classification/ionosphere.csv')

    df = pd.read_csv('data/classification/ionosphere.csv' ,delimiter=',', header=None)
    display(df.head())

    X = df.drop(columns=df.columns[-1]).to_numpy(dtype=np.float32)
    y = ((df[df.columns[-1]] =='b') + 1).to_numpy(dtype=np.int32)
    return y, X
  
  def wdbc(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'

    urllib.request.urlretrieve(url, 'data/classification/wdbc.csv')

    df = pd.read_csv('data/classification/wdbc.csv', header=None)
    df.replace({'M': 1}, inplace=True)
    df.replace({'B': 0}, inplace=True)

    display(df.head())
    X = df.iloc[:,2:].to_numpy(dtype=np.float32)
    y = df.iloc[:,1].to_numpy(dtype=np.int32)
    return y, X
  
  def wine(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
    urllib.request.urlretrieve(url, 'data/classification/wine.csv')

    col_names = ['cultivars','Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',\
                'Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity',\
                'Hue', 'OD280/OD315','Proline']

    df = pd.read_csv('data/classification/wine.csv', names=col_names)

    X = df.drop("cultivars",axis = 1).to_numpy(dtype=np.float32)
    y = df["cultivars"].to_numpy(dtype=np.int32)

    display(df.head())
    return y, X

  def coil_2000(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt'
    urllib.request.urlretrieve(url, 'data/classification/COIL2000/coil200_Xy_train.csv')

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticeval2000.txt'
    urllib.request.urlretrieve(url, 'data/classification/COIL2000/coil200_X_test.csv')

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/tictgts2000.txt'
    urllib.request.urlretrieve(url, 'data/classification/COIL2000/coil200_y_test.csv')

    df = pd.read_csv('data/classification/COIL2000/coil200_Xy_train.csv',delimiter='\t',header=None)
    X_train = df.iloc[:,:-1].to_numpy(dtype=np.float32)
    y_train = df.iloc[:,85].to_numpy(dtype=np.int32)

    display(df.head())

    df = pd.read_csv('data/classification/COIL2000/coil200_X_test.csv',delimiter='\t',header=None)
    X_test = df.to_numpy(dtype=np.float32)

    df = pd.read_csv('data/classification/COIL2000/coil200_y_test.csv',delimiter='\t',header=None)
    y_test = df.to_numpy(dtype=np.int32)
    return y_train, X_train, y_test, X_test

  
  def abalone(self):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
    urllib.request.urlretrieve(url, 'data/classification/abalone.csv')

    columns = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
    df = pd.read_csv('data/classification/abalone.csv', names = columns)
    df = one_hot(df, ['Sex'])

    display(df.head())
    X = df.drop('Rings',axis = 1).to_numpy(dtype=np.float32)
    y = df['Rings'].to_numpy(dtype=np.int32)
    return y, X







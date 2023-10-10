import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import pickle
from sklearn.impute import SimpleImputer
import warnings
from os.path import join
warnings.filterwarnings('ignore')

from tqdm import tqdm
from utils.timefeatures import time_features

vital_IDs = ['HeartRate' , 'SysBP' , 'DiasBP' , 'MeanBP' , 'RespRate' , 'SpO2' , 'Glucose' ,'Temp']
lab_IDs = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN'
          'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC']
eth_list = ['white', 'black', 'hispanic', 'asian', 'other']

def add_time_features(dates, timeenc=1, freq='h'):
    df_stamp = pd.DataFrame()
    df_stamp['date'] = pd.to_datetime(dates)
    if timeenc == 0:
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = df_stamp.drop(['date'], 1).values
    elif timeenc == 1:
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
    return data_stamp

def quantize_signal(signal, start, step_size, n_steps, value_column, charttime_column):
    quantized_signal = []
    # quantized_counts = np.zeros((n_steps,))
    l = start
    u = start + timedelta(hours=step_size)
    for _ in range(n_steps):
        signal_window = signal[value_column][(signal[charttime_column]>l) & (signal[charttime_column]<u)]
        quantized_signal.append(signal_window.mean())
       
        l = u
        u = l + timedelta(hours=step_size)
        
    return quantized_signal # , quantized_counts

def quantize_time(start, step_size, n_steps):
    dates = [
        start + timedelta(hours=step_size*i) \
            for i in range(n_steps)
    ]
    return add_time_features(dates, timeenc=1, freq='h')


def check_nan(A):
    A = np.array(A)
    nan_arr = np.isnan(A).astype(int)
    nan_count = np.count_nonzero(nan_arr)
    return nan_arr, nan_count


def forward_impute(x, nan_arr):
    x_impute = x.copy()
    first_value = 0
    while first_value<len(x) and nan_arr[first_value]==1:
        first_value += 1
    last = x_impute[first_value]
    for i,measurement in enumerate(x):
        if nan_arr[i]==1:
            x_impute[i] = last
        else:
            last = measurement
    return x_impute


def impute_lab(lab_data):
    imputer = SimpleImputer(strategy="mean")
    lab_data_impute = lab_data.copy()
    imputer.fit( lab_data.reshape((-1,lab_data.shape[1])) )
    for i,patient in enumerate(lab_data):
        for j,signal in enumerate(patient):
            nan_arr , nan_count = check_nan(signal)
            if nan_count!=len(signal):
                lab_data_impute[i,j,:] = forward_impute(signal, nan_arr)
    lab_data_impute = np.array( [imputer.transform(sample.T).T for sample in lab_data_impute] )
    return lab_data_impute

output_folder = './dataset/mimic_iii/'
vital_data = pd.read_csv(
    join(output_folder, "adult_icu_vital.gz") , 
    compression='gzip'
)#, nrows=5000)

#print("Vitals:\n", vital_data[0:20])
vital_data = vital_data.dropna(subset=['vitalid'])

lab_data = pd.read_csv(
    join(output_folder, "adult_icu_lab.gz"), 
    compression='gzip'
)#, nrows=5000)
#print("Labs:\n", lab_data[0:20])
lab_data = lab_data.dropna(subset=['label'])

icu_id = list(vital_data.icustay_id.unique())
sample_size = 100
if sample_size is not None:
    seed = 22891
    np.random.seed(seed)
    icu_id_subsampled = np.random.choice(icu_id, size=sample_size)
    icu_id = icu_id_subsampled

## features for every patient will be the list of vital IDs, gender(male=1, female=0), age, ethnicity(unknown=0 ,white=1, black=2, hispanic=3, asian=4, other=5), first_icu_stay(True=1, False=0)
x = np.zeros((len(icu_id), 12 , 48))
x_lab = np.zeros((len(icu_id), len(lab_IDs) , 48))
x_impute = np.zeros((len(icu_id), 12, 48))

time_encoded_features = add_time_features([pd.to_datetime('2020-01-01')]).shape[1]
x_time_features = np.zeros((len(icu_id), time_encoded_features, 48))
dates = np.zeros(len(icu_id))

y = np.zeros((len(icu_id),))
imp_mean = SimpleImputer(strategy="mean")

missing_ids = []
missing_map = np.zeros((len(icu_id), 12))
missing_map_lab = np.zeros((len(icu_id), len(lab_IDs)))
nan_map = np.zeros((len(icu_id), len(lab_IDs)+12))

for i,id in tqdm(enumerate(icu_id), total=len(icu_id), miniters=10):
    patient_data = vital_data.loc[vital_data['icustay_id']==id]
    patient_data['vitalcharttime'] = patient_data['vitalcharttime'].astype('datetime64[s]')
    patient_lab_data = lab_data.loc[lab_data['icustay_id']==id]
    patient_lab_data['labcharttime'] = patient_lab_data['labcharttime'].astype('datetime64[s]')
    eth_coder = lambda eth:0 if eth=='0' else eth_list.index(patient_data['ethnicity'].iloc[0])+1

    admit_time = patient_data['vitalcharttime'].min()
    #print('Patient %d admitted at '%(id),admit_time)
    n_missing_vitals = 0

    ## Extract demographics and repeat them over time
    x[i,-4,:]= int(patient_data['gender'].iloc[0])
    x[i,-3,:]= int(patient_data['age'].iloc[0])
    x[i,-2,:]= eth_coder(patient_data['ethnicity'].iloc[0])
    x[i,-1,:]= int(patient_data['first_icu_stay'].iloc[0])
    y[i] = (int(patient_data['mort_icu'].iloc[0]))
    
    x_time_features[i] = quantize_time(
        start=admit_time, step_size=1, n_steps=48
    ).T
    dates[i] = admit_time

    ## Extract vital measurement information
    # vitals = patient_data.vitalid.unique()
    for vital, signal in patient_data.groupby('vitalid'):
        try:
            id_index = vital_IDs.index(vital)
            # signal = patient_data[patient_data['vitalid']==vital]
            quantized_signal = quantize_signal(
                signal, start=admit_time, 
                step_size=1, n_steps=48, value_column='vitalvalue', 
                charttime_column='vitalcharttime'
            )
            nan_arr, nan_count = check_nan(quantized_signal)
            x[i, id_index] = np.array(quantized_signal)
            
            nan_map[i,len(lab_IDs)+vital_IDs.index(vital)] = nan_count
            if nan_count==48:
                n_missing_vitals =+ 1
                missing_map[i,vital_IDs.index(vital)]=1
            else:
                x_impute[i,:,:] = imp_mean.fit_transform(x[i,:,:].T).T
                  
        except:
            pass

    ## Extract lab measurement informations
    # labs = patient_lab_data.label.unique()
    for lab, lab_measures in patient_lab_data.groupby('label'):
        try:
            lab_index = lab_IDs.index(lab)
            # lab_measures = patient_lab_data[patient_lab_data['label']==lab]
            quantized_lab = quantize_signal(
                lab_measures, start=admit_time, step_size=1, 
                n_steps=48, value_column='labvalue', 
                charttime_column='labcharttime'
            )
            nan_arr, nan_count = check_nan(quantized_lab)
            x_lab[i, lab_index] = np.array(quantized_lab)
            nan_map[i, lab_index] = nan_count
            if nan_count == 48:
                missing_map_lab[i, lab_index]=1
        except:
            pass

    ## Remove a patient that is missing a measurement for the entire 48 hours
    if n_missing_vitals>0:
        missing_ids.append(i)
    if i == 100:
        break


## Record statistics of the dataset, remove missing samples and save the signals
f = open(join(output_folder, "stats.txt"), "a")
f.write('\n ******************* Before removing missing *********************')
f.write('\n Number of patients: '+ str(len(y))+'\n Number of patients who died within their stay: '+str(np.count_nonzero(y)))
f.write("\nMissingness report for Vital signals")
for i,vital in enumerate(vital_IDs):	
        f.write("\nMissingness for %s: %.2f"%(vital,np.count_nonzero(missing_map[:,i])/len(icu_id)))
        f.write("\n")
f.write("\nMissingness report for Vital signals")	
for i,lab in enumerate(lab_IDs):
        f.write("\nMissingness for %s: %.2f"%(lab,np.count_nonzero(missing_map_lab[:,i])/len(icu_id)))
        f.write("\n")

x = np.delete(x, missing_ids, axis=0)
x_lab = np.delete(x_lab, missing_ids, axis=0)
x_impute = np.delete(x_impute, missing_ids, axis=0)

x_time_features = np.delete(x_time_features, missing_ids, axis=0)
dates = np.delete(dates, missing_ids, axis=0)
y = np.delete(y, missing_ids, axis=0)
nan_map = np.delete(nan_map, missing_ids, axis=0)

x_lab_impute = impute_lab(x_lab)
missing_map = np.delete(missing_map, missing_ids, axis=0)
missing_map_lab = np.delete(missing_map_lab, missing_ids, axis=0)
all_data = np.concatenate((x_lab_impute, x_impute), axis=1)
print(f'All data shape {all_data.shape}, time features shape {x_time_features.shape}')

f.write('\n ******************* After removing missing *********************')
f.write('\n Final number of patients: '+str(len(y))+'\n Number of patients who died within their stay: '+str(np.count_nonzero(y)))
f.write("\nMissingness report for Vital signals")	
for i,vital in enumerate(vital_IDs):
        f.write("\nMissingness for %s: %.2f"%(vital,np.count_nonzero(missing_map[:,i])/len(icu_id)))
        f.write("\n")
f.write("\nMissingness report for Vital signals")	
for i,lab in enumerate(lab_IDs):
        f.write("\nMissingness for %s: %.2f"%(lab,np.count_nonzero(missing_map_lab[:,i])/len(icu_id)))
        f.write("\n")
f.close()

samples = [ (all_data[i,:,:],y[i],nan_map[i,:], x_time_features, dates) for i in range(len(y)) ]
with open(join(output_folder, 'patient_vital_preprocessed.pkl'),'wb') as f:
        pickle.dump(samples, f)
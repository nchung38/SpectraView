#-------------------------------------------------------------------------------------------------------
# Suite of functions either ported from existing matlab code or written for the ML workflow
# Brandon Chan | January 2021 | For spectra plasmonics
#-------------------------------------------------------------------------------------------------------
import os # For filesystem navigation
import pandas as pd # For dataframe/data manipulation
import numpy as np # For linear algebra and scietific computing
import numpy.matlib # Required specifically for some matrix multiplication operations
import scipy
from scipy import interpolate # Part of spectra pre-processing
from scipy.signal import savgol_filter # Part of spectra-preprocessing
#from sklearn.base import is_classifier
from time import strftime



def load_spectra(filename, mode=''):
    '''
    Generic data loader for spectra data. 
    - Determines type of loading depending on file extension. 
    - Assumes certian charateristics and formatting. 
    - Basic error checks built-in
    
    Inputs:
        filename = a string of the full filepath of the spectra to be loaded
    
    Output:
        data = pandas dataframe of the loaded spectra (Could convert to a numpy array if so desired)
    '''
    # Option 1. TXT. Assumes txt files are pre-formatted, header stripped, etc. (ie. same as other training files)
    if filename.split('.')[-1] == 'txt' and mode == 'nonformatted':
        data = pd.read_csv(filename, engine='python', sep=r'\s{1,}', header=8, names=['x','y'])
        data['x'] = data['x'].apply(lambda x: int(str(x).strip(',')))
    elif filename.split('.')[-1] == 'txt' and mode == 'ocean':
        data = pd.read_csv(filename, engine='python', sep=r'\s{1,}', header=12, names=['x','y'])
    elif filename.split('.')[-1] == 'txt':
        data = pd.read_csv(filename, engine='python', sep=r'\s{1,}', header=14, names=['x','y'])
    # Option 2: CSV. Assumes a csv exported from anton-paar with the standard pre-header row structure
    elif filename.split('.')[-1] == 'csv' and mode == 'nonformatted':
        data =  pd.read_csv(filename, header=8, names=['x','y'])
    elif filename.split('.')[-1] == 'csv':
        data =  pd.read_csv(filename)
    else:
        print("Unsupported Filetype")
        return

    # Error checks: Check dims and if columns were read wrong
    if len(data) == 0:
        print("No data in file:", filename)
        return
    elif np.any(np.isnan(data.mean().values)):
        print("Error reading data:", filename)
        return
    
    return data



def backcor(n ,y, odr, s, fct):
    '''
    Python implmentation of V. Mazet's backcorr function originally written in MATLAB.
    Please see https://www.mathworks.com/matlabcentral/fileexchange/27429-background-correction
    for more detailed explaination and documentation.
    
    Inputs:
        n = numpy array of shape [x,]
        y = numpy array of shape [x,]
        odr = integer
        s = float
        fct = string specifying 'sh', 'ah', 'stq', or 'atq'
    
    Outputs:
        z = numpy array of shape [x,]
        a =  numpy array of shape [ord,]
        it = integer
        odr = integer
        s = float
        fct = str
    '''
    N = len(n) 
    i = np.argsort(n)
    n = np.sort(n)
    
    y = y[i]
    
    maxy = max(y)
    
    dely = (maxy - min(y)) / 2
    n = 2 * (n[:] - n[N-1]) / (n[N-1] - n[0]) + 1
    
    y = (y[:] - maxy) / dely + 1
    
    p = np.arange(0,odr+1)
    
    T = np.power(np.matlib.repmat(n, odr+1, 1).transpose(), np.matlib.repmat(p, N, 1))
    
    Tinv = np.matmul(np.linalg.pinv(np.matmul(T.transpose(),T)), T.transpose())
    
    a = np.matmul(Tinv, y)
    z = np.matmul(T, a)
    
    alpha = 0.99 * 1/2
    it = 0               
    zp = np.ones(N)
    
    while sum((z-zp)**2)/sum(zp**2) > 1e-9:
        it = it + 1
        zp = z
        res = y - z
    
        if fct == 'sh':
            d = (res*(2*alpha-1)) * (abs(res)<s) + (-alpha*2*s-res) * (res<=-s) + (alpha*2*s-res) * (res>=s)
        elif fct == 'ah':
            d = (res*(2*alpha-1)) * (res<s) + (alpha*2*s-res) * (res>=s)
        elif fct == 'stq':
            d = (res*(2*alpha-1)) * (abs(res)<s) - res * (abs(res)>=s)
        elif fct == 'atq':
            d = (res*(2*alpha-1)) * (res<s) - res * (res>=s)
        
        a = np.matmul(Tinv, (y+d))  
        z = np.matmul(T,a)
    
    j = np.argsort(i)
    z = (z[j]-1)*dely + maxy

    a[0] = a[0]-1
    a = a*dely #% + maxy # Some residual comment that was left in the matlab code handed over. carried over to python implmentation
    
    return z, a, it, odr, s, fct


def SGfilter_Baseline_Int(x0, y0):
    '''
    Python implmentation of the SGfilter_Baseline_Int function written by Spectra Plasmonics
    
    (Essentially a wrapped call to backcor() with predetermined parameters.
     could consider removing and consolidating in preprocesing_pipeline())
    
    Inputs:
        x0 = numpy array of size [x,]
        y0 = numpy array of size [x,]
    
    Outputs:
        yzint = numpy array of size [x,]
    '''
    odr = 10
    s = 0.01
    fct = 'atq'
    z, _, _, _, _, _ = backcor(x0, y0, odr, s, fct)
    yzint = y0 - z
    return yzint


def apply_interpolation(x0, wave_numb, y_zint):
    '''
    Python implementation of the Int function written by Sepctra Plasmonics
    *changed function name to be more descripive
    
    (Essentially a wrapped code to call to an interpolation function. 
     could consider removing and consolidating in preprocesing_pipeline())
    
    Inputs:
        x0 = a numpy array of shape [x,]
        wave_numb = a numpy array of shape [y,]
        y_zint = a numpy array of shape [x,]

    Outputs:
        yzint = interpolated output
    '''
    tck = interpolate.splrep(x0, y_zint,  s=0)
    yzint = interpolate.splev(wave_numb, tck, der=0)
    
    return yzint


def normalization(spectra):
    '''
    Normalization on a per-spectra basis. Adapted from the MATLAB script "Normalize_Spectra.m"
    
    Inputs:
        spectra = numpy array of shape [n,] (Only reprents the data of one sample)
    
    Outputs:
        spec_norm = normalized spectra
    '''
    spec_mean = spectra.mean()
    rs_center = spectra - spec_mean
    
    rs_center_sq = rs_center**2
    rs_length = sum(rs_center_sq)**0.5
    
    spec_norm = rs_center / rs_length
    
    #spec_norm = spectra/np.max(spectra,axis=0)
    
    #spec_norm = spectra
    
    return spec_norm
    

def preprocesing_pipeline(data, **kwargs):
    '''
    Function that wraps the preprocessing steps together, accepting the data and required 
    processing parameters as input, returning a processed sample.
    
    Inputs: 
        data = pandas dataframe with 2 columns
        **kwargs:
            - SGpoly = integer
            - SGframe = integer
            - wave_numb = numpy array of shape [x,] (generated from np.linspace)
        
    Outputs:
        processed_spectra = numpy array the with shape [x,] where x is the dimension from wave_numb 
    '''     
    
    #data2 = data.iloc[1:]
    
    x0 = data['x'].values
    y0 = savgol_filter(data['y'].values, polyorder=kwargs['SGpoly'], window_length=kwargs['SGframe'])
    
    #lam = kwargs.get('Lambda',1.0*10**-3)
    #y = data['y'].values
    #L = len(y)
    #D = scipy.sparse.csc_matrix(np.diff(np.eye(L), 2))
    #w = np.ones(L)
    #W = scipy.sparse.spdiags(w, 0, L, L)
    #Z = W + lam * D.dot(D.transpose())
    #y0 = scipy.sparse.linalg.spsolve(Z, w*y)
    
    y_zint = SGfilter_Baseline_Int(x0, y0)
    yzint = apply_interpolation(x0, kwargs['wave_numb'], y_zint)
    
    processed_spectra = (yzint)
    
    return processed_spectra


def prepare_training_data(directory_path, file_type='csv', mode='', label_name='', **kwargs):
    '''
    Function that reads in spectra files from a specified directory, processes them per the preprocessing
    pipeline and returns two numpy arrays; one for the spectra themselves, and one for the corresponding labels
    
    Inputs:
        directory_path = string 
        label_name = string
        **kwargs:
            - SGpoly = integer
            - SGframe = integer
            - wlow = integer
            - whigh = integer
            - wave_numb = numpy array of shape [x,] (generated from np.linspace)
            
    Outputs:
        data_array = numpy array of shape [x, n] where x is the number of samples (rows) 
                     and n is the number of "features" (columns) 
        data_labels = numpy array of shape [x, ] that containes the corresponding labels
                      per row in data_array.
    '''
    # Load in filenames freom 
    files = os.listdir(directory_path)
    files_filtered = [x for x in files if x.split('.')[-1] == file_type]
    num_files = len(files_filtered)
    
    # Preallocate outputs
    data_array = np.empty((num_files, (kwargs['whigh'] - kwargs['wlow'] + 1)))
    data_array[:,:] = np.nan
    data_labels = np.empty(num_files, dtype=object)
    
    # Assign label name to labels output array
    data_labels[:] = label_name
    
    i = 0 
    for filename in files_filtered:
        if mode == 'nonformatted':
            input_data = load_spectra(directory_path + filename, mode='nonformatted')
        else:
            input_data = load_spectra(directory_path + filename)
        processed_data = preprocesing_pipeline(input_data, SGpoly=kwargs['SGpoly'], SGframe=kwargs['SGframe'], wave_numb=kwargs['wave_numb'])
        
        data_array[i, :] = processed_data
        i+=1

    return data_array, data_labels

def prepare_calibration_data(directory_path, file_type='csv', mode='', **kwargs):
    '''
    Function that reads in spectra files from a specified directory, processes them per the preprocessing
    pipeline and returns two numpy arrays; one for the spectra themselves, and one for the corresponding labels
    
    Inputs:
        directory_path = string 
        label_name = string
        **kwargs:
            - SGpoly = integer
            - SGframe = integer
            - wlow = integer
            - whigh = integer
            - wave_numb = numpy array of shape [x,] (generated from np.linspace)
            
    Outputs:
        data_array = numpy array of shape [x, n] where x is the number of samples (rows) 
                     and n is the number of "features" (columns) 
        data_labels = numpy array of shape [x, ] that containes the corresponding labels
                      per row in data_array.
    '''
    # Load in filenames freom 
    files = os.listdir(directory_path)
    files_filtered = [x for x in files if x.split('.')[-1] == file_type]
    num_files = len(files_filtered)
    
    # Preallocate outputs
    data_array = np.empty((num_files, (kwargs['whigh'] - kwargs['wlow'] + 1)))
    data_array[:,:] = np.nan
    
    # Assign label name to labels output array
   
    i = 0 
    for filename in files_filtered:
        if mode == 'nonformatted':
            input_data = load_spectra(directory_path + filename, mode='nonformatted')
        else:
            input_data = load_spectra(directory_path + filename)
        processed_data = preprocesing_pipeline(input_data, SGpoly=kwargs['SGpoly'], SGframe=kwargs['SGframe'], wave_numb=kwargs['wave_numb'])
        
        data_array[i, :] = processed_data
        i+=1

    return data_array

def prepare_training_data_ind_files(filenames, file_type='csv', mode='', label_name='', **kwargs):
    '''
    Function that reads in spectra files from a specified directory, processes them per the preprocessing
    pipeline and returns two numpy arrays; one for the spectra themselves, and one for the corresponding labels
    
    Inputs:
        directory_path = string 
        label_name = string
        **kwargs:
            - SGpoly = integer
            - SGframe = integer
            - wlow = integer
            - whigh = integer
            - wave_numb = numpy array of shape [x,] (generated from np.linspace)
            
    Outputs:
        data_array = numpy array of shape [x, n] where x is the number of samples (rows) 
                     and n is the number of "features" (columns) 
        data_labels = numpy array of shape [x, ] that containes the corresponding labels
                      per row in data_array.
    '''
    # Load in filenames freom 
    num_files = len(filenames)
    
    # Preallocate outputs
    data_array = np.empty((num_files, (kwargs['whigh'] - kwargs['wlow'] + 1)))
    data_array[:,:] = np.nan
    data_labels = np.empty(num_files, dtype=object)
    
    # Assign label name to labels output array
    data_labels[:] = label_name
    
    i = 0 
    for filename in filenames:
        if mode == 'nonformatted':
            input_data = load_spectra(filenames[i], mode='nonformatted')
        else:
            input_data = load_spectra(filenames[i])
        processed_data = preprocesing_pipeline(input_data, SGpoly=kwargs['SGpoly'], SGframe=kwargs['SGframe'], wave_numb=kwargs['wave_numb'])
        
        data_array[i, :] = processed_data
        i+=1

    return data_array, data_labels


def infer_class_label(predictions_df):
    '''
    Function to automate the extraction of the true class label based on the prefix or filename assigned to a
    spectra collected for the purpose of model development.
    
    Inputs:
        predictions_df = a pandas dataframe that should contain at the minimum a column called "filename"
                         that denotes the filename of the spectra
    Outputs:
        A pandas dataframe with two columns: filename and true, where true denotes the actual class label of
        the file
    '''
    labels = []
    
    for f in predictions_df.filename:
        tag = f[0]
        if tag == 'A':
            labels += [{'filename':f, 'true':'carfentanil'}] #'unknown'
        elif tag == 'B':
            labels += [{'filename':f, 'true':'eti'}]
        elif tag == 'C':
            labels += [{'filename':f, 'true':'fentanyl'}]
        elif tag == 'D':
            labels += [{'filename':f, 'true':'meth'}]
        elif tag == 'E':
            labels += [{'filename':f, 'true':'fen-meth'}]
        elif tag == 'F':
            labels += [{'filename':f, 'true':'feneti'}]
        elif tag == 'G':
            labels += [{'filename':f, 'true':'careti'}]    
        elif tag == 'H':
            labels += [{'filename':f, 'true':'non-detect'}] 
        else:
            print(tag, 'ERROR')
            
    return pd.DataFrame(labels)
    

def predict_on_directory(clf, pls, directory_path, infer_labels=True, **kwargs):
    '''
    Function to simplify the process of using a pre-trained model to predict a bunch of spectra stored in 
    any given directory.
    
    Inputs:
        clf = a fitted sklearn classifier (currently implied to be an SVC)
        pls = a fitted sklearn PLSRegression object
        directory_path = a string denoting the path to the target directory
        infer_labels = a boolean flag to trigger the extraction of the true class label based on the 
                       filename of the spectra. Default is True.
        **kwargs:
            - SGpoly = integer
            - SGframe = integer
            - wlow = integer
            - whigh = integer
            - wave_numb = numpy array of shape [x,] (generated from np.linspace)
    Outputs:
        A pandas dataframe containing the filename, date stamp, predicted class label, and probability output 
        of each potential class label. If infer_labels is True, an additional column will be present to denote 
        the true class label of the corresponding filename. Each row of the dataframe represents one sample 
        (from one file)
    '''
    if str(type(pls)) != "<class 'sklearn.cross_decomposition._pls.PLSRegression'>":
        raise TypeError("Input arg 'pls' must be of type sklearn.cross_decomposition._pls.PLSRegression. Please check input args.")
    if is_classifier(clf) == False:
        raise TypeError("Input arg 'clf' must be a scikit learn classifier. Please check input args")
        
    files = os.listdir(directory_path)
    files_filtered = [x for x in files if x.split('.')[-1] == 'csv']
    date_stamp = strftime("%d-%m-%Y %H:%M:%S")
    
    predictions = []
    
    for filename in files_filtered:
        input_data = load_spectra(directory_path + filename)#, mode='nonformatted')
        processed_data = preprocesing_pipeline(input_data, SGpoly=kwargs['SGpoly'], SGframe=kwargs['SGframe'], wave_numb=kwargs['wave_numb'])

        pls_features = pls.transform(processed_data.reshape(1, -1))

        prediction = clf.predict(pls_features)[0]
        prediction_probabilty = clf.predict_proba(pls_features)

        log_dict = {'filename':filename, 'date':date_stamp, 'predicted_substance':prediction, 'prob_predict':np.nan}
        i = 0
        for c in clf.classes_:
            log_dict['prob_'+c] = prediction_probabilty[0][i]

            if c == prediction:
                log_dict['prob_predict'] = prediction_probabilty[0][i]

            i += 1

        predictions += [log_dict]
    
    predictions = pd.DataFrame(predictions)
    
    if infer_labels:
        labels = infer_class_label(predictions)
        return pd.merge(predictions, labels)
    else:
        return predictions
    
    
    
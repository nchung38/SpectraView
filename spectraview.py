import numpy.matlib
import scipy
from scipy import interpolate # Part of spectra pre-processing
from scipy.signal import savgol_filter # Part of spectra-preprocessing
#from sklearn.base import is_classifier
from time import strftime
from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import font
from tkinter import StringVar
from tkinter import messagebox
from tkinter.messagebox import showerror
from tkinter.messagebox import showinfo
from tkinter.filedialog import asksaveasfilename
import os # For filesystem navigation
import pandas as pd # For dataframe/data manipulation
import numpy as np # For linear algebra and scietific computing
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
import sys
sys.path.insert(0,"../")
#from helper_functions import * 

matplotlib.use('TkAgg',force=True)

global header
header = 14

SGpoly = 3
SGframe = 9
wlow = 500
whigh = 3500
wspace = whigh - wlow + 1
wave = np.linspace(wlow, whigh, wspace)
wave_numb = wave.transpose() 
axesbool = 0
spectracount = 1
cal_spectracount = 1

root = Tk()
root.title("SpectraView")

UnderlinedFont = font.Font(family='Calibri', name='UnderlinedFont', size=12, underline = 1 )
TitleFont = font.Font(family='Helvetica', name='TitleFont', size=24, weight='bold' )

def load_spectra(filename, mode=''):
    '''
    Generic data loader for spectra data. 
    - Determines type of loading depending on file extension. 
    - Assumes certain characteristics and formatting. 
    - Basic error checks built-in
    
    Inputs:
        filename = a string of the full filepath of the spectra to be loaded
    
    Output:
        data = pandas dataframe of the loaded spectra (Could convert to a numpy array if so desired)
    '''
    # Option 1. TXT. Assumes txt files are pre-formatted, header stripped, etc. (ie. same as other training files)
    if filename.split('.')[-1] == 'txt':
        data = pd.read_csv(filename, engine='python', sep=r'\s{1,}', header=header, names=['x','y'])
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
    Python implimentation of V. Mazet's backcorr function originally written in MATLAB.
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
    if filtervar.get() == 1:
        y0 = savgol_filter(data['y'].values, polyorder=kwargs['SGpoly'], window_length=kwargs['SGframe'])
        y_zint = SGfilter_Baseline_Int(x0, y0)
        yzint = apply_interpolation(x0, kwargs['wave_numb'], y_zint)
    
    elif filtervar.get() == 0:
        y_zint = data['y'].values
        yzint = apply_interpolation(x0, kwargs['wave_numb'], y_zint)
    #lam = kwargs.get('Lambda',1.0*10**-3)
    #y = data['y'].values
    #L = len(y)
    #D = scipy.sparse.csc_matrix(np.diff(np.eye(L), 2))
    #w = np.ones(L)
    #W = scipy.sparse.spdiags(w, 0, L, L)
    #Z = W + lam * D.dot(D.transpose())
    #y0 = scipy.sparse.linalg.spsolve(Z, w*y)
    
       
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


def openfiles():
    filetypes = (
        ('text files', '*.txt'),
        ('All files', '*.*')
    )

    filenames = fd.askopenfilenames(
        title='Open files',
        initialdir='/',
        filetypes=filetypes)
    
    if filenames:
        wlow = (wlowint.get())    
        whigh = (whighint.get()) 
        wspace = whigh - wlow + 1
        wave = np.linspace(wlow, whigh, wspace)
        global wave_numb
        wave_numb = wave.transpose() 
        spectra_data, spectra_labels = prepare_training_data_ind_files(filenames,
                                                            file_type='txt',
                                                            SGpoly=SGpoly,
                                                            SGframe=SGframe,
                                                            wlow=wlow,
                                                            whigh=whigh,
                                                            wave_numb=wave_numb)
                                                            
        spectra_avg = (np.average(spectra_data, axis=0))
        assignspectra(spectra_avg)

def openfolder():
    file_path = fd.askdirectory()
    if file_path:    
        file_path = file_path + "\\"
        wlow = (wlowint.get())    
        whigh = (whighint.get()) 
        wspace = whigh - wlow + 1
        wave = np.linspace(wlow, whigh, wspace)
        global wave_numb
        wave_numb = wave.transpose() 
        spectra_data, spectra_labels = prepare_training_data(file_path,
                                                            file_type='txt',
                                                            SGpoly=SGpoly,
                                                            SGframe=SGframe,
                                                            wlow=wlow,
                                                            whigh=whigh,
                                                            wave_numb=wave_numb)
                                                            
        spectra_avg = (np.average(spectra_data, axis=0))
        assignspectra(spectra_avg)


def assignspectra(spectra_avg):
    global spectracount
    if spectracount == 1:
        global spectra1
        spectra1 = spectra_avg
        selectone.grid(column=0, row=5, padx = 5)
    if spectracount == 2:
        global spectra2
        spectra2 = spectra_avg
        selecttwo.grid(column=1, row=5, padx = 5)
    elif spectracount ==  3:
        global spectra3
        spectra3 = spectra_avg
        selectthree.grid(column=2, row=5, padx = 5)
    elif spectracount ==  4:
        global spectra4
        spectra4 = spectra_avg
        selectfour.grid(column=3, row=5, padx = 5)
    elif spectracount ==  5:
        global spectra5
        spectra5 = spectra_avg
        selectfive.grid(column=4, row=5, padx = 5)
    elif spectracount >=  6:
            messagebox.showinfo(message='Max amount of spectra saved. Please clear spectra to save more.')
    spectracount = spectracount + 1


ca_combo_count = 1
def assigntocombo_ca():
    global ca_combo_count

    yabbadabbado = (spectraselect.get())
    if yabbadabbado == "one":
        add_spectra = spectra1
        add_name = selectone["text"]
    if yabbadabbado == "two":
        add_spectra = spectra2
        add_name = selecttwo["text"]
    if yabbadabbado ==  "three":
        add_spectra = spectra3
        add_name = selectthree["text"]
    if yabbadabbado ==  "four":
        add_spectra = spectra4
        add_name = selectfour["text"]
    if yabbadabbado ==  "five":
        add_spectra = spectra5
        add_name = selectfive["text"]
    ca_dict[add_name] = (add_spectra,)

    ca_combo['values'] = (*ca_combo['values'],add_name)

    ca_combo.grid(column = 1, row = 2)
    select_ca.grid(column = 0, row = 2,padx = 5,pady =5,sticky = "W")

her_combo_count = 1
def assigntocombo_her():
    global her_combo_count

    yabbadabbado = (spectraselect.get())
    if yabbadabbado == "one":
        add_spectra = spectra1
        add_name = selectone["text"]
    if yabbadabbado == "two":
        add_spectra = spectra2
        add_name = selecttwo["text"]
    if yabbadabbado ==  "three":
        add_spectra = spectra3
        add_name = selectthree["text"]
    if yabbadabbado ==  "four":
        add_spectra = spectra4
        add_name = selectfour["text"]
    if yabbadabbado ==  "five":
        add_spectra = spectra5
        add_name = selectfive["text"]
    her_dict[add_name] = (add_spectra,)

    her_combo['values'] = (*her_combo['values'],add_name)

    her_combo.grid(column = 1, row = 3)
    select_her.grid(column = 0, row = 3,padx = 5,pady =5,sticky = "W")
    
ki_combo_count = 1
def assigntocombo_ki():
    global ki_combo_count

    yabbadabbado = (spectraselect.get())
    if yabbadabbado == "one":
        add_spectra = spectra1
        add_name = selectone["text"]
    if yabbadabbado == "two":
        add_spectra = spectra2
        add_name = selecttwo["text"]
    if yabbadabbado ==  "three":
        add_spectra = spectra3
        add_name = selectthree["text"]
    if yabbadabbado ==  "four":
        add_spectra = spectra4
        add_name = selectfour["text"]
    if yabbadabbado ==  "five":
        add_spectra = spectra5
        add_name = selectfive["text"]
    ki_dict[add_name] = (add_spectra,)

    ki_combo['values'] = (*ki_combo['values'],add_name)


    ki_combo.grid(column = 1, row = 4)
    select_ki.grid(column = 0, row = 4,padx = 5,pady =5,sticky = "W")


def plotspectra():

    yabbadabbado = (spectraselect.get())
    if yabbadabbado == "one":
        plotted_spectra = spectra1
        plottedname = selectone["text"]
    if yabbadabbado == "two":
        plotted_spectra = spectra2
        plottedname = selecttwo["text"]
    if yabbadabbado ==  "three":
        plotted_spectra = spectra3
        plottedname = selectthree["text"]
    if yabbadabbado ==  "four":
        plotted_spectra = spectra4
        plottedname = selectfour["text"]
    if yabbadabbado ==  "five":
        plotted_spectra = spectra5
        plottedname = selectfive["text"]
    if yabbadabbado ==  "ca":
        plottedname = ca_select.get()
        plotted_spectra = (ca_dict[plottedname])[0]
    if yabbadabbado ==  "her":
        plottedname = her_select.get()
        plotted_spectra = (her_dict[plottedname])[0]    
    if yabbadabbado ==  "ki":
        plottedname = ki_select.get()
        plotted_spectra = (ki_dict[plottedname])[0]

    specplot.plot(wave_numb,plotted_spectra, label = plottedname)
    #axes = plt.axes()
    specplot.legend()

    specplot.set_xlabel('Raman Shift ($cm^{-1}$)')
    specplot.set_ylabel('Intensity')
    specplot.set_xlim([np.amin(wave_numb), np.amax(wave_numb)])

    canvas.draw()


baselinecount = 1

def addbaseline():
    file_path = fd.askdirectory()
    if file_path:
        file_path = file_path + "\\"
        wlow = (wlowint.get())    
        whigh = (whighint.get()) 
        wspace = whigh - wlow + 1
        wave = np.linspace(wlow, whigh, wspace)
        global wave_numb
        wave_numb = wave.transpose() 
        spectra_data = prepare_calibration_data(file_path,
                                                            file_type='txt',
                                                            SGpoly=SGpoly,
                                                            SGframe=SGframe,
                                                            wlow=wlow,
                                                            whigh=whigh,
                                                            wave_numb=wave_numb)
        global baselinespectra                                                     
        baselinespectra = np.average(spectra_data,axis=0)
        global baselinecount
        baselinecount = 2
        return baselinespectra

def plotbaseline():
    global baselinecount
    if baselinecount == 1:

        wlow = (wlowint.get())    
        whigh = (whighint.get()) 
        wspace = whigh - wlow + 1
        wave = np.linspace(wlow, whigh, wspace)
        global wave_numb
        wave_numb = wave.transpose() 
        spectra_data, spectra_labels = prepare_training_data(r"C:\Users\Student003\Downloads\Nick\SERS statistical analysis\2.6 PCB (HBS) Experiment - 50 nm\With GNU\Mother Files\Interaction\RS\\",
                                                             file_type='txt',
                                                             SGpoly=SGpoly,
                                                             SGframe=SGframe,
                                                             wlow=wlow,
                                                             whigh=whigh,
                                                             wave_numb=wave_numb)
                                                         
        global baselinespectra
        baselinespectra = (np.average(spectra_data, axis=0))
        baselinecount = 2
    
        specplot.plot(wave_numb,baselinespectra, label = "Baseline")
        specplot.legend()
        #axes = plt.axes()
        specplot.set_xlabel('Raman Shift ($cm^{-1}$)')
        specplot.set_ylabel('Intensity')
        specplot.set_xlim([np.amin(wave_numb), np.amax(wave_numb)])
        canvas.draw()

    elif baselinecount == 2:
        specplot.plot(wave_numb,baselinespectra,label = "Baseline")
        specplot.legend()
        axes = plt.axes()
        specplot.set_xlabel('Raman Shift ($cm^{-1}$)')
        specplot.set_ylabel('Intensity')
        specplot.set_xlim([np.amin(wave_numb), np.amax(wave_numb)])
        canvas.draw()
    
        
def renamespectra():
    takename = renamename.get()
    bingbong = (spectraselect.get())    
    if bingbong == "one":
        selectone["text"] = takename
    if bingbong == "two":
        selecttwo["text"] = takename
    if bingbong ==  "three":
        selectthree["text"] = takename
    if bingbong ==  "four":
        selectfour["text"] = takename
    if bingbong ==  "five":
        selectfive["text"] = takename
  


def clearspectra():
    global spectracount
    spectracount = 1
    spectra1 = 0
    spectra2 = 0
    spectra3 = 0
    spectra4 = 0
    spectra5 = 0
    selectone.grid_forget()
    selecttwo.grid_forget()
    selectthree.grid_forget()
    selectfour.grid_forget()
    selectfive.grid_forget()

def clearset():
    yabbadabbado = (spectraselect.get())
    if yabbadabbado ==  "ca":
        ca_combo['values'] = []
        ca_dict = {}
        ca_combo.grid_forget()
        select_ca.grid_forget()
    if yabbadabbado ==  "her":
        her_combo['values'] = []
        her_dict = {}    
        her_combo.grid_forget()
        select_her.grid_forget()
    if yabbadabbado ==  "ki":
        ki_combo['values'] = []
        ki_dict = {}
        ki_combo.grid_forget()
        select_ki.grid_forget()

def clearplot():
    specplot.clear()
    canvas.draw()


def savefile():
    bingbong = (spectraselect.get())
    if bingbong == "one":
        saved_spectra = spectra1
    if bingbong == "two":
        saved_spectra = spectra2
    if bingbong ==  "three":
        saved_spectra = spectra3
    if bingbong ==  "four":
        saved_spectra = spectra4
    if bingbong ==  "five":
        saved_spectra = spectra5
    if bingbong ==  "ca":
        savedname = ca_select.get()
        saved_spectra = (ca_dict[savedname])[0]
    if bingbong ==  "her":
        savedname = her_select.get()
        saved_spectra = (her_dict[savedname])[0]    
    if bingbong ==  "ki":
        savedname = ki_select.get()
        saved_spectra = (ki_dict[savedname])[0]

    files = [('All Files', '*.*'), 
             ('Excel Files', '*.xlsx')]
    savefilename = asksaveasfilename(filetypes=(("Excel files", "*.xlsx"),
                                                          ("All files", "*.*") ))
    spectra_df = pd.DataFrame({'Raman Shift':wave_numb, 'Intensity':saved_spectra})
    spectra_df.to_excel(savefilename + ".xlsx", index = False)


def addcalibrationfiles():
    file_path = fd.askdirectory()
    if file_path:
        file_path = file_path + "\\"
        wlow = (wlowint.get())    
        whigh = (whighint.get()) 
        wspace = whigh - wlow + 1
        wave = np.linspace(wlow, whigh, wspace)
        global wave_numb
        wave_numb = wave.transpose() 
        spectra_data = prepare_calibration_data(file_path,
                                                            file_type='txt',
                                                            SGpoly=SGpoly,
                                                            SGframe=SGframe,
                                                            wlow=wlow,
                                                            whigh=whigh,
                                                            wave_numb=wave_numb)
                                                            
        assign_cal_spectra(spectra_data)

def assign_cal_spectra(spectra_data):
    global cal_spectracount
    if cal_spectracount == 1:
        global cal_spectra1
        cal_spectra1 = spectra_data
        concentration_label1.grid(column=1, row=5, padx = 5)
        concentration_entry1.grid(column=2, row=5, padx = 5)
    elif cal_spectracount == 2:
        global cal_spectra2
        cal_spectra2 = spectra_data
        concentration_label2.grid(column=1, row=6, padx = 5)
        concentration_entry2.grid(column=2, row=6, padx = 5)
    elif cal_spectracount ==  3:
        global cal_spectra3
        cal_spectra3 = spectra_data
        concentration_label3.grid(column=1, row=7, padx = 5)
        concentration_entry3.grid(column=2, row=7, padx = 5)
    elif cal_spectracount ==  4:
        global cal_spectra4
        cal_spectra4 = spectra_data
        concentration_label4.grid(column=1, row=8, padx = 5)
        concentration_entry4.grid(column=2, row=8, padx = 5)
    elif cal_spectracount ==  5:
        global cal_spectra5
        cal_spectra5 = spectra_data
        concentration_label5.grid(column=1, row=9, padx = 5)
        concentration_entry5.grid(column=2, row=9, padx = 5)
    elif cal_spectracount ==  6:
        global cal_spectra6
        cal_spectra6 = spectra_data
        concentration_label6.grid(column=1, row=10, padx = 5)
        concentration_entry6.grid(column=2, row=10, padx = 5)
    elif cal_spectracount ==  7:
        global cal_spectra7
        cal_spectra7 = spectra_data
        concentration_label7.grid(column=1, row=11, padx = 5)
        concentration_entry7.grid(column=2, row=11, padx = 5)
    elif cal_spectracount ==  8:
        global cal_spectra8
        cal_spectra8 = spectra_data
        concentration_label8.grid(column=1, row=12, padx = 5)
        concentration_entry8.grid(column=2, row=12, padx = 5)
    elif cal_spectracount ==  9:
        global cal_spectra9
        cal_spectra9 = spectra_data
        concentration_label9.grid(column=1, row=13, padx = 5)
        concentration_entry9.grid(column=2, row=13, padx = 5)
    elif cal_spectracount ==  10:
        global cal_spectra10
        cal_spectra10 = spectra_data
        concentration_label10.grid(column=1, row=14, padx = 5)
        concentration_entry10.grid(column=2, row=14, padx = 5)
    elif cal_spectracount >= 11:
        messagebox.showinfo(message='Max amount of calibration spectra.')
    cal_spectracount = cal_spectracount + 1

def addtestfiles():
    file_path = fd.askdirectory()
    if file_path:
        file_path = file_path + "\\"
        wlow = (wlowint.get())    
        whigh = (whighint.get()) 
        wspace = whigh - wlow + 1
        wave = np.linspace(wlow, whigh, wspace)
        global wave_numb
        wave_numb = wave.transpose() 
        spectra_data = prepare_calibration_data(file_path,
                                                            file_type='txt',
                                                            SGpoly=SGpoly,
                                                            SGframe=SGframe,
                                                            wlow=wlow,
                                                            whigh=whigh,
                                                            wave_numb=wave_numb)
        global test_spectra                                                     
        test_spectra = spectra_data
        return test_spectra

def createcurve():
    try:
        if cal_spectracount >= 2:
            root.setvar(name ="concentration_int1", value = int(conc1.get()))
        if cal_spectracount >= 3:
            root.setvar(name ="concentration_int2", value = int(conc2.get()))
        if cal_spectracount >= 4:
            root.setvar(name ="concentration_int3", value = int(conc3.get()))
        if cal_spectracount >= 5:
            root.setvar(name ="concentration_int4", value = int(conc4.get()))
        if cal_spectracount >= 6:   
            root.setvar(name ="concentration_int5", value = int(conc5.get()))
        if cal_spectracount >= 7:
            root.setvar(name ="concentration_int6", value = int(conc6.get()))
        if cal_spectracount >= 8:
            root.setvar(name ="concentration_int7", value = int(conc7.get()))
        if cal_spectracount >= 9:
            root.setvar(name ="concentration_int8", value = int(conc8.get()))
        if cal_spectracount >= 10:
            root.setvar(name ="concentration_int9", value = int(conc9.get()))
        if cal_spectracount >= 11:
            root.setvar(name ="concentration_int10", value = int(conc10.get()))


    except ValueError as error:
        showerror(title='Error', message=error)

    global calibration_spectra
    global calibration_conc

    
    if cal_spectracount == 1:
        messagebox.showinfo(message='Please add calibration spectra to create curve.')

    if cal_spectracount >= 2:
        
        calibration_spectra = cal_spectra1
        conc1_array = np.array([])
        for i in range(len(cal_spectra1)):
            conc1_array = np.append(conc1_array, concentration_int1.get())
        
        calibration_conc = conc1_array

    if cal_spectracount >= 3:
        
        calibration_spectra = np.concatenate([calibration_spectra,cal_spectra2])
        conc2_array = np.array([])
        for i in range(len(cal_spectra2)):
            conc2_array = np.append(conc2_array, concentration_int2.get())
        
        calibration_conc = np.concatenate([calibration_conc, conc2_array])
        
    if cal_spectracount >=  4:
        calibration_spectra = np.concatenate([calibration_spectra, cal_spectra3])
        conc3_array = np.array([])

        for i in range(len(cal_spectra3)):
            conc3_array = np.append(conc3_array, concentration_int3.get())
        
        calibration_conc = np.concatenate([calibration_conc, conc3_array])

    if cal_spectracount >=  5:
        calibration_spectra = np.concatenate([calibration_spectra, cal_spectra4])
        conc4_array = np.array([])

        for i in range(len(cal_spectra4)):
            conc4_array = np.append(conc4_array, concentration_int4.get())
        
        calibration_conc = np.concatenate([calibration_conc, conc4_array])

    if cal_spectracount >=  6:
        calibration_spectra = np.concatenate([calibration_spectra, cal_spectra5])
        conc3_array = np.array([])

        for i in range(len(cal_spectra5)):
            conc5_array = np.append(conc5_array, concentration_int5.get())
        
        calibration_conc = np.concatenate([calibration_conc, conc5_array])

    if cal_spectracount >=  7:
        calibration_spectra = np.concatenate([calibration_spectra, cal_spectra6])
        conc6_array = np.array([])

        for i in range(len(cal_spectra6)):
            conc6_array = np.append(conc6_array, concentration_int6.get())
        
        calibration_conc = np.concatenate([calibration_conc, conc6_array])

    if cal_spectracount >=  8:
        calibration_spectra = np.concatenate([calibration_spectra, cal_spectra7])
        conc7_array = np.array([])

        for i in range(len(cal_spectra7)):
            conc7_array = np.append(conc3_array, concentration_int7.get())
        
        calibration_conc = np.concatenate([calibration_conc, conc7_array])

    if cal_spectracount >=  9:
        calibration_spectra = np.concatenate([calibration_spectra, cal_spectra8])
        conc8_array = np.array([])

        for i in range(len(cal_spectra8)):
            conc8_array = np.append(conc8_array, concentration_int8.get())
        
        calibration_conc = np.concatenate([calibration_conc, conc8_array])

    if cal_spectracount >=  10:
        calibration_spectra = np.concatenate([calibration_spectra, cal_spectra9])
        conc9_array = np.array([])

        for i in range(len(cal_spectra8)):
            conc9_array = np.append(conc9_array, concentration_int9.get())
        
        calibration_conc = np.concatenate([calibration_conc, conc9_array])

    if cal_spectracount >=  11:
        calibration_spectra = np.concatenate([calibration_spectra, cal_spectra10])
        conc10_array = np.array([])

        for i in range(len(cal_spectra10)):
            conc10_array = np.append(conc10_array, concentration_int10.get())
        
        calibration_conc = np.concatenate([calibration_conc, conc10_array])

    X = calibration_spectra
    Y = calibration_conc

    n_comp = 20
    mse = []
    component = np.arange(1, n_comp)
 
    for i in component:
        pls = PLSRegression(n_components=i)
 
        # Cross-validation
        y_cv = cross_val_predict(pls, X, Y, cv=3)
 
        mse.append(mean_squared_error(Y, y_cv))
 
    global msemin
    msemin = np.argmin(mse)

    pls_opt = PLSRegression(n_components=msemin+1)
    pls_opt.fit(X, Y)
    Z = pls_opt.predict(X)

    global ymax
    ymax = np.amax(Y)
    global ymin
    ymin = np.amin(Y)

    poly = np.polyfit(Y,Z,1)

    ycheck = Y[0]
    XX = np.array([])
    YY = np.array([])
    countx = 0
    county = 1
    YY = np.append(YY,ycheck)
    for (i) in (Y):
        if i == ycheck:
            countx += 1
        else:
            XX = np.append(XX,countx)
            countx = 1
            ycheck = i
            county += 1
            YY = np.append(YY,i)

    XX = np.append(XX,countx)
    tempz = np.array([])
    ZZ = np.array([])

    countz = 0
    countagain = 0
    for i in (YY):
        for j in range(int((XX[countz]))):
            tempz = np.append(tempz,Z[countagain])
            countagain += 1
        addz = np.average(tempz)
        tempz = np.array([])
        ZZ = np.append(ZZ,addz)
        countz += 1
                


    curveplot.clear()
    curveplot.scatter(YY,ZZ)
    curveplot.plot(np.polyval(poly,Y),Y, linewidth = 1)
    curveplot.set_xlabel('Actual Concentration')
    curveplot.set_ylabel('Predicted Concentration')
    curveplot.set_title('Calibration Curve')
    curvecanvas.draw()

def plotunknownspectra():

    X = calibration_spectra
    Y = calibration_conc

    pls_opt = PLSRegression(n_components=msemin+1)
    pls_opt.fit(X, Y)
    Zpredictions = pls_opt.predict(test_spectra)

    Z = np.average(Zpredictions)
    zpercent = int(threshold)*(0.2)

    if Z > (threshold + zpercent):
        redlight()
    elif Z < (threshold + zpercent) and Z > (threshold - zpercent):
        orangelight()
    elif Z < (threshold - zpercent):
        greenlight()
    
    threshlinex = np.linspace(ymin,ymax, num = 50) 
    threshliney = []
    for i in threshlinex:
        threshliney.append(threshold)

    curveplot.scatter(Z,Z)
    curveplot.plot(threshlinex,threshliney)
    curveplot.set_xlabel('Actual Concentration')
    curveplot.set_ylabel('Predicted Concentration')
    curveplot.set_title('Prediction Curve')
    curvecanvas.draw()

def redlight():
    orangelighttext.grid_forget()
    greenlighttext.grid_forget()
    redlighttext.grid(row = 5, column = 0,pady = 20)
    
def orangelight():
    redlighttext.grid_forget()
    greenlighttext.grid_forget()
    orangelighttext.grid(row = 5, column = 0,pady = 20)
    
def greenlight():
    redlighttext.grid_forget()
    orangelighttext.grid_forget()
    greenlighttext.grid(row = 5, column = 0,pady = 20)

def clearcurve():
    curveplot.clear()
    #curveplot.plot()
    curvecanvas.draw()


left_frame = Frame(root, background='grey')
left_frame.grid(row=0, column=0, padx=10, pady=5, sticky = (N,W))
right_frame = Frame(root, background='white')
right_frame.grid(row=0, column=1, sticky = (E))


specname = ""
spectra_name = ttk.Label(left_frame, text = specname)
spectra_name.grid(row=0,column=0, sticky = (N,W))

content = ttk.Frame(left_frame, padding=(3,3,12,12))
content.grid(row=0,column=0, sticky = (N))
content['relief'] = 'sunken'

buttonbox = ttk.Frame(left_frame)
buttonbox.grid(row=1,column=0, sticky = (N))
buttonbox['relief'] = 'sunken'

buttonbox2 = ttk.Frame(left_frame)
buttonbox2.grid(row=2,column=0, sticky = (N,E,W))
buttonbox2['relief'] = 'sunken'

add_to_ca_button = ttk.Button(buttonbox, text='Add to CA-15-3 Set')
add_to_ca_button.grid(column=1, row=1, padx = 5, pady = 10 )
add_to_ca_button.configure(command=assigntocombo_ca)

add_to_her_button = ttk.Button(buttonbox, text='Add to HER-II Set')
add_to_her_button.grid(column=2, row=1, padx = 5, pady = 10 )
add_to_her_button.configure(command=assigntocombo_her)

add_to_ca_button = ttk.Button(buttonbox, text='Add to KI-67 Set')
add_to_ca_button.grid(column=3, row=1, padx = 5, pady = 10 )
add_to_ca_button.configure(command=assigntocombo_ki)

ca_select = tk.StringVar()
ca_combo = ttk.Combobox(buttonbox, textvariable=ca_select)
ca_combo['values'] = []
ca_combo['state'] = 'readonly'
ca_dict = {}

her_select = tk.StringVar()
her_combo = ttk.Combobox(buttonbox, textvariable=her_select)
her_combo['values'] = []
her_combo['state'] = 'readonly'
her_dict = {}

ki_select = tk.StringVar()
ki_combo = ttk.Combobox(buttonbox, textvariable=ki_select)
ki_combo['values'] = []
ki_combo['state'] = 'readonly'
ki_dict = {}


global selecttext_one
global selecttext_two
global selecttext_three
global selecttext_four
global selecttext_five
global selecttext_ca
selecttext_one = "One"
selecttext_two = "Two"
selecttext_three = "Three"
selecttext_four = "Four"
selecttext_five = "Five"
selecttext_ca = "CA-15-3"
selecttext_her = "HER-II"
selecttext_ki = "KI-67"



spectraselect = tk.StringVar()
selectone = ttk.Radiobutton(buttonbox, text=selecttext_one, variable=spectraselect, value='one')
selecttwo = ttk.Radiobutton(buttonbox, text=selecttext_two, variable=spectraselect, value='two')
selectthree = ttk.Radiobutton(buttonbox, text=selecttext_three, variable=spectraselect, value='three')
selectfour = ttk.Radiobutton(buttonbox, text=selecttext_four, variable=spectraselect, value='four')
selectfive = ttk.Radiobutton(buttonbox, text=selecttext_five, variable=spectraselect, value='five')        
select_ca = ttk.Radiobutton(buttonbox, text=selecttext_ca, variable=spectraselect, value='ca')
select_her = ttk.Radiobutton(buttonbox, text=selecttext_her, variable=spectraselect, value='her')
select_ki = ttk.Radiobutton(buttonbox, text=selecttext_ki, variable=spectraselect, value='ki')



concentration_int1 = tk.IntVar(root, name = "concentration_int1")
concentration_int2 = tk.IntVar(root,name = "concentration_int2")
concentration_int3 = tk.IntVar(root,name = "concentration_int3")
concentration_int4 = tk.IntVar(root,name = "concentration_int4")
concentration_int5 = tk.IntVar(root,name = "concentration_int5")
concentration_int6 = tk.IntVar(root,name = "concentration_int6")
concentration_int7 = tk.IntVar(root,name = "concentration_int7")
concentration_int8 = tk.IntVar(root,name = "concentration_int8")
concentration_int9 = tk.IntVar(root,name = "concentration_int9")
concentration_int10 = tk.IntVar(root,name = "concentration_int10")
conc1 = tk.StringVar()
conc2 = tk.StringVar()
conc3 = tk.StringVar()
conc4 = tk.StringVar()
conc5 = tk.StringVar()
conc6 = tk.StringVar()
conc7 = tk.StringVar()
conc8 = tk.StringVar()
conc9 = tk.StringVar()
conc10 = tk.StringVar()
concentration_entry1 = ttk.Entry(buttonbox2, textvariable = conc1)
concentration_entry2 = ttk.Entry(buttonbox2, textvariable = conc2)
concentration_entry3 = ttk.Entry(buttonbox2, textvariable = conc3)
concentration_entry4 = ttk.Entry(buttonbox2, textvariable = conc4)
concentration_entry5 = ttk.Entry(buttonbox2, textvariable = conc5)
concentration_entry6 = ttk.Entry(buttonbox2, textvariable = conc6)
concentration_entry7 = ttk.Entry(buttonbox2, textvariable = conc7)
concentration_entry8 = ttk.Entry(buttonbox2, textvariable = conc8)
concentration_entry9 = ttk.Entry(buttonbox2, textvariable = conc9)
concentration_entry10 = ttk.Entry(buttonbox2, textvariable = conc10)
concentration_label1 = ttk.Label(buttonbox2, text = "Concentration 1:")
concentration_label2 = ttk.Label(buttonbox2, text = "Concentration 2:")
concentration_label3 = ttk.Label(buttonbox2, text = "Concentration 3:")
concentration_label4 = ttk.Label(buttonbox2, text = "Concentration 4:")
concentration_label5 = ttk.Label(buttonbox2, text = "Concentration 5:")
concentration_label6 = ttk.Label(buttonbox2, text = "Concentration 6:")
concentration_label7 = ttk.Label(buttonbox2, text = "Concentration 7:")
concentration_label8 = ttk.Label(buttonbox2, text = "Concentration 8:")
concentration_label9 = ttk.Label(buttonbox2, text = "Concentration 9:")
concentration_label10 = ttk.Label(buttonbox2, text = "Concentration 10:")


screenw = (root.winfo_screenwidth())
screenh = (root.winfo_screenheight())

wid = screenw/150
hei = screenh/150

figframe = plt.Figure(figsize=(wid, hei))
specplot = figframe.add_subplot(111)
canvas = FigureCanvasTkAgg(figframe, right_frame)
canvas.get_tk_widget().grid(column = 0, row = 0, sticky = (S,E))
specplot.grid()
canvas.draw()

curveframe = plt.Figure(figsize=(wid/2, hei/2))
curveplot = curveframe.add_subplot(111)
curvecanvas = FigureCanvasTkAgg(curveframe, right_frame)
curvecanvas.get_tk_widget().grid(column = 0, row = 1, sticky = (N),pady = 5)
curveframe.subplots_adjust(bottom=0.15)
curveplot.grid()
curvecanvas.draw()


redlighttext = ttk.Label(left_frame, text = "Caution" , background = "red")
orangelighttext = ttk.Label(left_frame, text = "Alert", background = "orange")
greenlighttext = ttk.Label(left_frame, text = "Normal", background = "green")

#buffer = ttk.Label(right_frame, text = "")
#buffer.grid(column =0, row = 3)

wlowint = IntVar(root, name ="wlowint")
root.setvar(name ="wlowint", value = 500)
whighint = IntVar(root, name ="whighint")
root.setvar(name ="whighint", value = 3500)

wlowstring = tk.StringVar()
wlow_entry1 = ttk.Entry(content, textvariable=wlowstring)
wlow_entry1.grid(column=2, row=8, )

whighstring = tk.StringVar()
whigh_entry1 = ttk.Entry(content, textvariable=whighstring)
whigh_entry1.grid(column=2, row=9,)

wlowlabel = ttk.Label(content, text = "Raman Shift Lower Limit")
whighlabel = ttk.Label(content, text = "Raman Shift Upper Limit")
wlowlabel.grid(column=1, row=8)
whighlabel.grid(column=1, row=9)

def setwave():

    try:
        root.setvar(name ="wlowint", value = int(wlowstring.get()))
        root.setvar(name ="whighint", value = int(whighstring.get()))


    except ValueError as error:
        showerror(title='Error', message=error)

thresholdint = IntVar(root, name ="thresholdint")

def setthreshold():

    try:
        root.setvar(name ="thresholdint", value = int(threshold_string.get()))
        global threshold
        threshold = thresholdint.get()
    except ValueError as error:
        showerror(title='Error', message=error)


rename_button = ttk.Button(content, text = "Rename Selected Spectra")
rename_button.grid(column=2,row = 0, sticky = "E")
rename_button.configure(command = renamespectra)

renamename = tk.StringVar()
rename_entry = ttk.Entry(content, textvariable = renamename)
rename_entry.grid(column=2, row = 1, sticky = "E")

wave_button = ttk.Button(content, text='Set Limits')
wave_button.grid(column=3, row=9, sticky='W', )
wave_button.configure(command=setwave)

plot_button = ttk.Button(buttonbox, text='Plot')
plot_button.grid(column=1, row=0, sticky='W', pady = 10, padx = 5)
plot_button.configure(command=plotspectra)

clear_spectra_button = ttk.Button(buttonbox, text = "Clear Saved Spectra")
clear_spectra_button.grid(column=2, row = 0, padx =5)
clear_spectra_button.configure(command=clearspectra)

clear_set_button = ttk.Button(buttonbox, text = "Clear Selected Set")
clear_set_button.grid(column=3, row = 0, padx = 5)
clear_set_button.configure(command = clearset)

add_baseline_button = ttk.Button(content, text = "Add Baseline")
add_baseline_button.grid(column = 0, row = 3, padx = 5, pady = 5, sticky = "W")
add_baseline_button.configure(command = addbaseline)

baseline_button = ttk.Button(content, text = "Plot Baseline")
baseline_button.grid(column = 0, row = 4, padx = 5, sticky = "W")
baseline_button.configure(command = plotbaseline)

add_calibration_button = ttk.Button(buttonbox2, text = "Add Calibration Files")
add_calibration_button.grid(column = 0, row = 0, padx = 10, pady = 2, sticky = "W")
add_calibration_button.configure(command =addcalibrationfiles)

add_test_button = ttk.Button(buttonbox2, text = "Add Files to Analyze")
add_test_button.grid(column = 0, row =1, padx = 10, pady = 2, sticky = "W")
add_test_button.configure(command = addtestfiles)

curve_button = ttk.Button(buttonbox2, text = "Calibrate Curve")
curve_button.grid(column = 0, row = 2, padx = 10, pady = 2, sticky = "W")
curve_button.configure(command = createcurve)

#plot_curve_button = ttk.Button(buttonbox2, text = "Plot Curve")
#plot_curve_button.grid(column = 0, row = 2, padx = 10, pady = 2, sticky = "W")
#plot_curve_button.configure(command = plotcalibrationcurve)

plot_predictions_button = ttk.Button(buttonbox2, text = "Plot Predictions")
plot_predictions_button.grid(column = 1, row = 2, padx = 10, pady = 2, sticky = "E")
plot_predictions_button.configure(command = plotunknownspectra)

set_thresh_button = ttk.Button(buttonbox2, text = "Set Threshold")
set_thresh_button.grid(column = 1, row =3)
set_thresh_button.configure(command = setthreshold)

threshold_string = tk.StringVar()
threshold_entry = ttk.Entry(buttonbox2, textvariable = threshold_string)
threshold_entry.grid(column = 2, row = 3, pady = 10)

clear_curve_button = ttk.Button(buttonbox2, text = "Clear Curve Plot")
clear_curve_button.grid(column = 2, row = 2, padx = 10, pady = 2, sticky = "E")
clear_curve_button.configure(command = clearcurve)

openfile = ttk.Button(content, text="Open File", command = openfiles)
openfolderbtn = ttk.Button(content, text="Open Folder", command = openfolder)
clearplotbtn = ttk.Button(content, text="Clear Plot", command = clearplot)
saveas = ttk.Button(content, text="Save As", command = savefile)
titlelabel = ttk.Label(right_frame, text = "SpectraView", font = TitleFont )
titlelabel.grid(row = 0, column=0, sticky = (N,E))

filtervar = tk.IntVar()
filter = ttk.Checkbutton(content, text="Filter", variable=filtervar, onvalue=1, offvalue=0)
filter.grid(row = 5, column =0, pady = 10)

content.grid(column=0, row=0, sticky=(N, S, E, W))
buttonbox.grid(column=0, row=1, sticky=(N, S, E, W))
openfile.grid(column=0, row=0, columnspan=2, sticky=(N, W),padx=5)
openfolderbtn.grid(column=0, row=1, columnspan=2, sticky=(N, W),padx=5)
clearplotbtn.grid(column=0, row=2, columnspan=2, sticky=(N,W), padx=5)
saveas.grid(column=0, row =0, columnspan=2, sticky=(N,W), pady=0, padx=100)


root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)
root.rowconfigure(0, weight=1)

content.columnconfigure(0, weight=3)
content.columnconfigure(1, weight=3)
content.columnconfigure(2, weight=3)
content.columnconfigure(3, weight=1)
content.columnconfigure(4, weight=1)
content.rowconfigure(1, weight=1)

root.mainloop()
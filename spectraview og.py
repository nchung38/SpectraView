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

#import openpyxl
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

import time
import sys
sys.path.insert(0,"../")
from helper_functions import * 

matplotlib.use('TkAgg',force=True)

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

def openfiles():
    filetypes = (
        ('text files', '*.txt'),
        ('All files', '*.*')
    )

    filenames = fd.askopenfilenames(
        title='Open files',
        initialdir='/',
        filetypes=filetypes)
    
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
    file_path = fd.askdirectory() + "\\"
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
        selectone.grid(column=0, row=2, padx = 5)
    if spectracount == 2:
        global spectra2
        spectra2 = spectra_avg
        selecttwo.grid(column=1, row=2, padx = 5)
    elif spectracount ==  3:
        global spectra3
        spectra3 = spectra_avg
        selectthree.grid(column=2, row=2, padx = 5)
    elif spectracount ==  4:
        global spectra4
        spectra4 = spectra_avg
        selectfour.grid(column=3, row=2, padx = 5)
    elif spectracount ==  5:
        global spectra5
        spectra5 = spectra_avg
        selectfive.grid(column=4, row=2, padx = 5)
    elif spectracount >=  6:
            messagebox.showinfo(message='Max amount of spectra saved. Please clear spectra to save more.')
    spectracount = spectracount + 1

def plotspectra():

    yabbadabbado = (spectraselect.get())

    if yabbadabbado == "one":
        plotted_spectra = spectra1
    if yabbadabbado == "two":
        plotted_spectra = spectra2
    if yabbadabbado ==  "three":
        plotted_spectra = spectra3
    if yabbadabbado ==  "four":
        plotted_spectra = spectra4
    if yabbadabbado ==  "five":
        plotted_spectra = spectra5
            
    specplot.plot(wave_numb,plotted_spectra)
    #axes = plt.axes()
    
    specplot.set_xlabel('Raman Shift (cm^-1)')
    specplot.set_ylabel('Intensity')
    specplot.set_xlim([np.amin(wave_numb), np.amax(wave_numb)])

    canvas.draw()
    #specname = file_path
    #spectra_name = ttk.Label(right_frame, background = "white", text = specname, font = UnderlinedFont)
    #spectra_name.grid(row=0,column=0, sticky = (N,W))


baselinecount = 1

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
    
        specplot.plot(wave_numb,baselinespectra)
        #axes = plt.axes()
        specplot.set_xlabel('Raman Shift (cm^-1)')
        specplot.set_ylabel('Intensity')
        specplot.set_xlim([np.amin(wave_numb), np.amax(wave_numb)])
        canvas.draw()

    elif baselinecount == 2:
        specplot.plot(wave_numb,baselinespectra)
        axes = plt.axes()
        specplot.set_xlabel('Raman Shift (cm^-1)')
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

    files = [('All Files', '*.*'), 
             ('Excel Files', '*.xlsx')]
    savefilename = asksaveasfilename(filetypes=(("Excel files", "*.xlsx"),
                                                          ("All files", "*.*") ))
    spectra_df = pd.DataFrame({'Raman Shift':wave_numb, 'Intensity':saved_spectra})
    spectra_df.to_excel(savefilename + ".xlsx", index = False)


def addcalibrationfiles():
    file_path = fd.askdirectory() + "\\"
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
    file_path = fd.askdirectory() + "\\"
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

def plotcalibrationcurve():

    X = calibration_spectra
    Y = calibration_conc

    pls_opt = PLSRegression(n_components=msemin+1)
    pls_opt.fit(X, Y)
    Z = pls_opt.predict(X)

    poly = np.polyfit(Y,Z,1)

    specplot.clear()
    specplot.scatter(Y,Z)
    specplot.plot(np.polyval(poly,Y),Y, linewidth = 1)
    specplot.set_xlabel('Actual Concentration')
    specplot.set_ylabel('Predicted Concentration')
    specplot.set_title('Calibration Curve')
    canvas.draw()

def plotunknownspectra():

    X = calibration_spectra
    Y = calibration_conc

    pls_opt = PLSRegression(n_components=msemin+1)
    pls_opt.fit(X, Y)
    Z = pls_opt.predict(test_spectra)

    specplot.clear()
    specplot.bar(Y,Z)
    specplot.set_xlabel('Actual Concentration')
    specplot.set_ylabel('Predicted Concentration')
    specplot.set_title('Calibration Curve')
    canvas.draw()

left_frame = Frame(root, background='grey')
left_frame.grid(row=0, column=0, padx=10, pady=5, sticky = (N,W))
right_frame = Frame(root, background='white')
right_frame.grid(row=0, column=1, sticky = (S,E))


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

global selecttext_one
global selecttext_two
global selecttext_three
global selecttext_four
global selecttext_five
selecttext_one = "One"
selecttext_two = "Two"
selecttext_three = "Three"
selecttext_four = "Four"
selecttext_five = "Five"

spectraselect = tk.StringVar()
selectone = ttk.Radiobutton(buttonbox, text=selecttext_one, variable=spectraselect, value='one')
selecttwo = ttk.Radiobutton(buttonbox, text=selecttext_two, variable=spectraselect, value='two')
selectthree = ttk.Radiobutton(buttonbox, text=selecttext_three, variable=spectraselect, value='three')
selectfour = ttk.Radiobutton(buttonbox, text=selecttext_four, variable=spectraselect, value='four')
selectfive = ttk.Radiobutton(buttonbox, text=selecttext_five, variable=spectraselect, value='five')        


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
canvas.get_tk_widget().grid(column = 0, row = 1, sticky = (S,E))
specplot.grid()
canvas.draw()

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
plot_button.grid(column=1, row=1, sticky='W', pady = 10, padx = 5)
plot_button.configure(command=plotspectra)

clear_spectra_button = ttk.Button(buttonbox, text = "Clear Saved Spectra")
clear_spectra_button.grid(column=2, row = 1, padx =5)
clear_spectra_button.configure(command=clearspectra)

baseline_button = ttk.Button(content, text = "Plot Baseline")
baseline_button.grid(column = 0, row = 3, padx = 5, sticky = "W")
baseline_button.configure(command = plotbaseline)

add_calibration_button = ttk.Button(buttonbox2, text = "Add Calibration Files")
add_calibration_button.grid(column = 0, row = 0, padx = 10, pady = 2, sticky = "W")
add_calibration_button.configure(command =addcalibrationfiles)

add_test_button = ttk.Button(buttonbox2, text = "Add Files to Analyze")
add_test_button.grid(column = 0, row =1, padx = 10, pady = 2, sticky = "W")
add_test_button.configure(command = addtestfiles)

curve_button = ttk.Button(buttonbox2, text = "Calculate Curve")
curve_button.grid(column = 1, row = 1, padx = 10, pady = 2, sticky = "E")
curve_button.configure(command = createcurve)

plot_curve_button = ttk.Button(buttonbox2, text = "Plot Curve")
plot_curve_button.grid(column = 0, row = 2, padx = 10, pady = 2, sticky = "W")
plot_curve_button.configure(command = plotcalibrationcurve)

plot_predictions_button = ttk.Button(buttonbox2, text = "Plot Predictions")
plot_predictions_button.grid(column = 1, row = 2, padx = 10, pady = 2, sticky = "E")
plot_predictions_button.configure(command = plotunknownspectra)

# add padding to the frame and show it
#frame.grid(padx=10, pady=10)

openfile = ttk.Button(content, text="Open File", command = openfiles)
openfolderbtn = ttk.Button(content, text="Open Folder", command = openfolder)
clearplotbtn = ttk.Button(content, text="Clear Plot", command = clearplot)
saveas = ttk.Button(content, text="Save As", command = savefile)
titlelabel = ttk.Label(right_frame, text = "SpectraView", font = TitleFont )
titlelabel.grid(row = 0, column=0, sticky = (N,E))

#filter = ttk.Checkbutton(content, text="Filter")
#two = ttk.Checkbutton(content, text="Two", variable=twovar, onvalue=True)
#three = ttk.Checkbutton(content, text="Three", variable=threevar, onvalue=True)
#ok = ttk.Button(content, text="Okay")
#cancel = ttk.Button(content, text="Cancel")


content.grid(column=0, row=0, sticky=(N, S, E, W))
buttonbox.grid(column=0, row=1, sticky=(N, S, E, W))
#frame.grid(column=3, row=1, columnspan=3, rowspan=2, sticky=(N, S, E, W))
openfile.grid(column=0, row=0, columnspan=2, sticky=(N, W),padx=5)
openfolderbtn.grid(column=0, row=1, columnspan=2, sticky=(N, W),padx=5)
clearplotbtn.grid(column=0, row=2, columnspan=2, sticky=(N,W), pady=5, padx=5)
saveas.grid(column=0, row =0, columnspan=2, sticky=(N,W), pady=0, padx=100)
#filter.grid(column=0, row=3)

#ok.grid(column=3, row=4)
#cancel.grid(column=4, row=4)


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
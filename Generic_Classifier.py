# import tensorflow as tf
# import matplotlib.pyplot as plt
# from tensorflow.examples.tutorials.mnist import input_data
# import numpy as np
# import os
# import tf_basics as tfb
# from tkinter import filedialog
# from tkinter import *
from classes import tkinter_app
#from classes import dataset
from classes import datalogger
from classes import dataset
import json_api
import threading
import time


# #Datalogger Configuration
# datalogger_obj = datalogger.datalogger(r"C:\Datalog\Datalog.csv")
# #Read JSON Files for Dataset
# json_filename = r"D:\Python\Tensorflow\Neural_Networks\Image Classifier\Generic_Classifier\main.json"
# json_obj = json_api.read_json(json_filename)
# #Dataset Configuration
# dataset_obj = dataset.dataset(json_obj, datalogger_obj)
# dataset_obj.load_images()



def main():
    
    gui = tkinter_app.tkinter_app()
    window_title  = 'Loading Images'
    threading.Thread(target=gui.progressbar_app, args=(window_title,)).start()
    for i in range(101):
        gui.progress_value = i
        gui.progress_label_value = str(i) + ' of '+str(100)
        time.sleep(0.1)

    print("Complete")
    return
    
        
main()
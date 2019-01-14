# import tensorflow as tf
# import matplotlib.pyplot as plt
# from tensorflow.examples.tutorials.mnist import input_data
# import numpy as np
# import os
# import tf_basics as tfb
# from tkinter import filedialog
# from tkinter import *
# from classes import tkinter_app
from classes import datalogger, dataset, neurons, mqtt_api, tkinter_app
import json_api
import threading, time, os


#Datalogger Configuration
datalogger_obj = datalogger.datalogger(r"C:\Datalog\Datalog.csv")

def dataset_testcases():
    #Test Case 1 - Folder Dataset
    #test_dataset(r"D:\Python\Tensorflow\Neural_Networks\Image Classifier\Generic_Classifier\Config\Test JSON\Dataset\Dataset_Folder.json")
    #Test Case 2 - ZIP Dataset (already tested)
    #test_dataset(r"D:\Python\Tensorflow\Neural_Networks\Image Classifier\Generic_Classifier\Config\Test JSON\Dataset\Dataset_ZIP.json")
    #Test Case 3 - Google Images Dataset
    #test_dataset(r"D:\Python\Tensorflow\Neural_Networks\Image Classifier\Generic_Classifier\Config\Test JSON\Dataset\Dataset_GoogleImageDownload.json")
    return

def test_dataset(json_filename):
    #Read JSON Files for Dataset
    json_obj = json_api.read_json(json_filename)
    #Dataset Configuration
    dataset_obj = dataset.dataset(json_obj,datalogger_obj)
    dataset_obj.load_images()
    return dataset_obj

def neuron_testcases():
    base_path = r"D:\Python\Tensorflow\Neural_Networks\Image Classifier\Generic_Classifier\Config\Test JSON\Neurons"
    json_file = ["mnist_1.0_softmax.json", "mnist_2.0_five_layers_sigmoid.json","mnist_2.0_five_layers_relu.json",
                 "mnist_2.1_five_layers_relu_lrdecay.json","mnist_2.2_five_layers_relu_lrdecay_dropout.json",
                 "mnist_3.1_convolutional_bigger_dropout.json"]
    test_neurons(os.path.join(base_path,json_file[5]), 'e-visual')
    
    return

def test_neurons(json_filename, mode = 'train'):
    #Read JSON Files for Dataset
    json_obj = json_api.read_json(json_filename)
    #Dataset Configuration
    dataset_obj = test_dataset(json_filename)
    #Neuron Configuration
    neuron_obj = neurons.neurons(json_obj,dataset_obj,datalogger_obj)
    if mode == 'train':
        neuron_obj.build_graph()
        neuron_obj.execute_session()
    elif mode == 'e-visual':
        neuron_obj.embed_images_for_visualization()
    elif mode =='predict':
        neuron_obj.build_graph()
        neuron_obj.load_session()
        #Write code here to predict images from Camera or Video feed (eye.class)
    return

def tkinter_testcases():
    #test_progressbar()
    #test_debug_console()
    test_browsepopup()
    return


def test_progressbar():
    gui = tkinter_app.tkinter_app()
    window_title  = 'Loading Images'
    progress_thread = threading.Thread(target=gui.progressbar_app, args=(window_title,))
    progress_thread.start()
    count = 100
    for i in range(count):
        start_time = time.time()
        time.sleep(0.025)
        gui.progress_value = (i/(count-1))*100
        gui.progress_label_value = str(i+1) + ' of '+str(count-1)+ ' ETA - '+ str(int((time.time() - start_time)*(100-i)))+ ' seconds'
    del gui
    return

def test_debug_console():
    gui = tkinter_app.tkinter_app()
    window_title  = 'Debug Log'
    progress_thread = threading.Thread(target=gui.debug_console_app, args=(window_title,))
    progress_thread.start()
    return

def test_browsepopup():
    gui = tkinter_app.tkinter_app()
    window_title  = 'Select Datasource'
    gui.browse_popup_app(window_title)
    return

def mqtt_testcases():
    test_mqtt()
    return

def test_mqtt(json_filename):
    json_obj = json_api.read_json(json_filename)
    mqtt_obj = mqtt_api.mqtt_api(json_obj, datalogger_obj)
    mqtt_obj.mqtt_publish(type = 'sensors', name = 'tsense', payload = "Hello")
    return

def main():
    #dataset_testcases()
    
    neuron_testcases()
    
    #tkinter_testcases()
    
    return        

main()
datalogger_obj.destroy_debugconsole()
print("Complete")
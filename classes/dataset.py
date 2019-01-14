import time, os, threading
import cv2
import numpy as np
import zipfile
import progressbar
import inspect
import json
from classes import tkinter_app
#from PIL import Image
#import matplotlib.pyplot as plt
from skimage import io, transform
#from keras.preprocessing.image import ImageDataGenerator
from google_images_download import google_images_download
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

class dataset():
    
    def __init__(self, json_object, datalogger_obj):
        '''
        "dataset": {
                  "data_source": "",
                  "label_names" : "",
                    "points": "100",
                    "summary_steps" : 100,
                    "x_resolution": 50,
                    "y_resolution": 50
                   } 
        '''        
        self.load_settings_from_json(json_object)
        self.data_dict = {'train_X':[],'train_Y':[],'test_X':[],'test_Y':[]}
        self.data_folder = ''
        self.set_logger(datalogger_obj)
        return
    
    def toprint(self, printcontent):
        if(True):
            print ('{}'.format(printcontent))
            
    def set_logger(self, datalogger_obj = ''):
        #If datalogger object is passed store it & enable datalogging
        self.is_datalog_enabled = False
        if not datalogger_obj == '':
            self.is_datalog_enabled = True
            self.datalogger_obj = datalogger_obj
            
    def log(self, log_string):
        #This functions logs incoming variables to datalog with timestamp
        if self.is_datalog_enabled:
            category = self.__class__.__name__
            sub_category = inspect.stack()[1][3]
            self.datalogger_obj.set_log(category, sub_category, log_string)   
        self.toprint(log_string)
        return
    
    def load_settings_from_json(self, json_object):
        #Read all settings from JSON file
        self.data_source = json_object['dataset']['data_source']
        if self.data_source.upper() == "MNIST":
            self.data_source = "MNIST"
            self.label_names = [str(x) for x in range(10)]
            print(self.label_names)
            self.x_resolution, self.y_resolution = 28, 28
        else:
            self.label_names = json_object['dataset']['label_names'] 
            self.x_resolution = int(json_object['dataset']['x_resolution'])
            self.y_resolution = int(json_object['dataset']['y_resolution'])
            
        
        
        self.fraction = 0.8         
        return
        
    def load_images(self):
        #This function loads the images depending on the data source
        if self.data_source == "":
            self.browse_json_gui = tkinter_app.tkinter_app()
            window_title  = 'Select Datasource'
            self.data_source = self.browse_json_gui.browse_popup_app(window_title)
        elif self.data_source == "MNIST":
            self.load_default_dataset()
        elif ('.json' in self.data_source):
            self.get_images_from_web()
        else:
            #Check if labels are empty
            if(self.label_names == []):
                warning = "Missing label_names, please input label names to train"
                self.log(warning)
            else:
                self.log('Data Source - '+ self.data_source)
                if(os.path.isdir(self.data_source)):
                    self.data_folder = self.data_source
                    self.load_images_from_folder()
                elif ('.zip' in self.data_source):
                    self.load_images_from_zip()
                
        #Get the summary of Dataset
        self.dataset_summary()
        return
    
    def dataset_summary(self):
        #prints the summary of the images
        total_images = len(self.data_dict['train_X'])+len(self.data_dict['test_X'])
        self.log(str(total_images)+" images loaded Successfully!!!")
        return

    def reshaped_image(self, image):
        #This function reshapes the input image to x_resolution(int) & y_resolution(int) & 3 Channels(RGB)
        return transform.resize(image,(self.x_resolution , self.y_resolution , 1)) # (cols (width), rows (height)) and don't use np.resize()
    
    def train_test_split(self,train_data, train_labels, fraction):
        self.log('Fraction - '+ str(fraction))
        index = int(len(train_data)*fraction)
        return train_data[:index], train_labels[:index], train_data[index:], train_labels[index:]
    
    def next_train_batch(self, size):
        shape = self.data_dict['train_X'].shape[0]
        indices = np.random.permutation(shape)
        return (self.data_dict['train_X'][indices[:size]],self.data_dict['train_Y'][indices[:size]])
    
    def next_test_batch(self, size):
        shape = self.data_dict['test_X'].shape[0]
        indices = np.random.permutation(shape)
        return (self.data_dict['test_X'][indices[:size]],self.data_dict['test_Y'][indices[:size]])
    
    def load_images_from_folder(self):
        #This function categorizes the data_folder items as per label_names and creates respective one hot vectors
        #Inputs - Obj, label_names(list of strings)
        #Return - Stores Train & Test data as obj.data_dict{}
        #Check if the images are folderized as per label or if the label information is contained in the name 
        #of the image itself        
        self.log("Loading Images......")
        while(len(os.listdir(self.data_folder)) ==1 ):
            self.data_folder = os.path.join(self.data_folder,os.listdir(self.data_folder)[0])
        self.log('Data Folder - '+ self.data_folder)        
        
        data_folder_list = os.listdir(self.data_folder)
        is_image_folderized = (len(data_folder_list)==len(self.label_names))
        
        #If the images are folderized as per label categorize & assign one hot vector as per their folder name
        if(is_image_folderized):
            
            self.label_names = data_folder_list
            len_label_names = len(self.label_names)
            for folder in data_folder_list:
                l = [ [0, 1][folder.find(self.label_names[x]) != -1] for x in range(len_label_names) ]
                image_folder = os.path.join(self.data_folder, folder)
                Images =  os.listdir(image_folder)
                for image in Images:
                    path = os.path.join(image_folder, image)
                    img = cv2.imread(path) # @UndefinedVariable
                    self.data_dict['train_X'].append(self.reshaped_image(img))
                    self.data_dict['train_Y'].append(l)
        #Otherwise loop through every image in the folder & look for label names in theor file name. And use the file name to create 
        #one hot vector for the looped images
        else:
            #for image in data_folder_list:
            bar = progressbar.ProgressBar()
            self.open_progressbar()
            len_label_names = len(self.label_names)
            no_of_images = len(data_folder_list)
            for i in bar(range(no_of_images)):
                start_time = time.time()
                image = data_folder_list[i] #Image name
                path = os.path.join(self.data_folder, image) #Image path
                #One hot vector
                l = [ [0, 1][image.find(self.label_names[x]) != -1] for x in range(len_label_names) ] 
                img = cv2.imread(path) # @UndefinedVariable
                self.data_dict['train_X'].append(self.reshaped_image(img))
                self.data_dict['train_Y'].append(l)
                #Update progressbar
                self.update_progressbar(i, no_of_images,start_time)
            self.close_progressbar()       
        #Split train/test data usually 80% train 20%test  
        #Convert all items to np.array for ease of use
        for item in self.data_dict:
            self.data_dict[item] = np.array(self.data_dict[item])     
        self.data_dict['train_X'], self.data_dict['train_Y'], self.data_dict['test_X'], self.data_dict['test_Y'] = self.train_test_split(self.data_dict['train_X'], self.data_dict['train_Y'], self.fraction)    
        self.log("Loading Complete!!!")
        return
    
    def load_images_from_zip(self):
        self.log("Extracting Dataset....")
        archive = zipfile.ZipFile(self.data_source, 'r')
        self.data_folder = os.path.split(self.data_source)
        self.data_folder = os.path.join( self.data_folder[0], '_'.join(self.label_names))
        archive.extractall(self.data_folder)
        self.log("Extraction Complete!!! \n ")
        #Then load the images from the newly created folder
        self.load_images_from_folder()
        #imgdata = archive.read('train/cat.0.jpg')
        #print(archive.namelist())
        #print(imgdata)
        return 
    
    def get_images_from_web(self):
        #This function downloads images from Google using googleimagesdownload library using you Web Browser
        #Inputs - Obj.data_folder(send as .cng file), the function will automatically download the image to Downloads folder in Windows
        #If you want it to download to different location, change the downloads folder of you default browser
        
        args_list = ["keywords", "keywords_from_file", "prefix_keywords", "suffix_keywords",
             "limit", "format", "color", "color_type", "usage_rights", "size",
             "exact_size", "aspect_ratio", "type", "time", "time_range", "delay", "url", "single_image",
             "output_directory", "image_directory", "no_directory", "proxy", "similar_images", "specific_site",
             "print_urls", "print_size", "print_paths", "metadata", "extract_metadata", "socket_timeout",
             "thumbnail", "language", "prefix", "chromedriver", "related_images", "safe_search", "no_numbering",
             "offset"]
        self.log("Parsing Config File....")
        records = []
        json_file = json.load(open(self.data_source))
        for record in range(0,len(json_file['Records'])):
            arguments = {}
            for i in args_list:
                arguments[i] = None
            for key, value in json_file['Records'][record].items():
                arguments[key] = value
            self.label_names.append(arguments['keywords'])
            records.append(arguments)
        self.log("Parsing Conmplete!!!")
        #Assign datafolder as the same path as JSON Config file 
        self.data_folder = os.path.split(self.data_source)
        #Create a subfoler using all the label names to store the images
        self.data_folder = os.path.join(self.data_folder[0], '_'.join(self.label_names))
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)
        
        #Download Images from Web for each keywords mentioned in JSON file
        self.log("Downloading Dataset....")
        response = google_images_download.googleimagesdownload()
        for argument in records:
            absolute_image_paths = response.download(argument)
            #Create a folder per keyword
            label_foldername = os.path.join(self.data_folder,argument['keywords'])
            if not os.path.exists(label_foldername):
                os.mkdir(label_foldername)
            #move all the downloaded files to label name folder
            for path in absolute_image_paths[argument['keywords']]:
                filedir = os.path.split(path)
                filename = filedir[1]
                new_path = os.path.join(label_foldername,filename)
                if not os.path.exists(new_path):
                    os.rename(path,new_path)
        self.log("Download Complete\n....")
        #Then load the images from the newly created folder
        self.load_images_from_folder()
        return

    def load_default_dataset(self):
        # Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
        self.log('Default Dataset(MNIST)')
        mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
        self.data_dict['train_X'], self.data_dict['train_Y'] = mnist.train.images, mnist.train.labels
        self.data_dict['test_X'], self.data_dict['test_Y'] = mnist.test.images, mnist.test.labels  

    def open_progressbar(self):
        self.progress_gui = tkinter_app.tkinter_app()
        window_title  = 'Extracting & Resizing Images'
        threading.Thread(target=self.progress_gui.progressbar_app, args=(window_title,)).start()
        return
    
    def update_progressbar(self, current_index, total, start_time):
        #This function updated the progress bar
        time_remaining = int((time.time() - start_time)*(total-current_index))
        if time_remaining < 60:
            time_remaining = str(time_remaining)+ ' sec'
        else:
            quotient= time_remaining//60
            remainder=time_remaining%60
            time_remaining = str(quotient)+ ' minutes and '+str(remainder)+' seconds'
        progress_label_value = str(current_index+1)+ ' of '+ str(total)+ ' | ETA - '+ time_remaining
        progress_value = ((current_index+1)/total)*100
        self.progress_gui.progress_value = progress_value
        self.progress_gui.progress_label_value = progress_label_value
        return
    
    def close_progressbar(self):
        del self.progress_gui
# from classes import datalogger
# from classes import dataset
# import json_api
# 
# 
# #Datalogger Configuration
# datalogger_obj = datalogger.datalogger(r"C:\Datalog\Datalog.csv")
# #Read JSON Files for Dataset
# json_filename = r"D:\Python\Tensorflow\Neural_Networks\Image Classifier\Generic_Classifier\main.json"
# json_obj = json_api.read_json(json_filename)
# #Dataset Configuration
# dataset_obj = dataset.dataset(json_obj, datalogger_obj)
# dataset_obj.load_images()

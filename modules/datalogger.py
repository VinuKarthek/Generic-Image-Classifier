'''
Created on Oct 14, 2018

@author: Vinu Karthek
'''
import os, datetime, threading
from classes import tkinter_app


def init(datalog_path):
    #Initialize the class with Datalog file path
    max_file_size = 0
    #Check if datalog path is a file or a directory
    if not os.path.isdir(datalog_path):
        datalog_filepath = datalog_path #If datalog_path is file assign it to datalog_filepath variable
        datalog_dir = os.path.split(datalog_filepath)[0] #Extract datalog directory from datalog filepath
        check_if_folder_exists(datalog_dir) #Create datalog directory if it doesn't exists
    else:
        datalog_dir = datalog_path
        check_if_folder_exists(datalog_dir) #Create datalog directory if it doesn't exists
        datalog_filepath = os.path.join(datalog_dir,('log_'+get_time()+".csv"))
    print(get_time())
    open()
    return

def check_if_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return

def get_file_size():
    #Get Datalog File size in Bytes
    return os.path.getsize(datalog_filepath)

def get_time():
    #Returns Timestamp in MM-DD-YYYY-HH.MM format
    now = datetime.datetime.now()
    return str(now.strftime("%m-%d-%Y-%H.%M.%S"))

def get_log(length):
    #Returns N lines from Datalog, where N is Specified by Variable 'length'
    line = datalog_fileref.readlines()
    return line

def set_log(, category, sub_category, log_string):
    #Logs the incoming entires (Category, Subcategory, String) with timestamp
    #Category = calss_name = __class__.__name__
    #Subcategory = inspect.getframeinfo(inspect.currentframe()).function
    timestamp = get_time()
    line = category + ',' + sub_category + ',' + log_string + "," + timestamp +'\n'
    datalog_fileref.writelines(line)
    return

def log_execution_time():
    #Returns the execution time on the module for logging
    return

def open():
    datalog_fileref = open(datalog_filepath,'a+')
    line = 'category' + ',' + 'sub_category' + ',' + 'log_string' + "," + 'timestamp' +'\n'
    datalog_fileref.writelines(line)
    return

def close():
    datalog_fileref.close()
    return

def show_logger():
    #Separate thread to display & use queue to refresh datalog
    logger_gui = tkinter_app.tkinter_app()
    window_title  = 'Datalogger'
    threading.Thread(target=logger_gui.progressbar_app, args=(window_title,)).start()
    return
    
        
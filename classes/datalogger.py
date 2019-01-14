'''
Created on Oct 6, 2018

@author: Vinu Karthek
'''

import os, datetime, threading
from classes import tkinter_app
import logging
try:
    import tkinter as tk # Python 3.x
    import tkinter.scrolledtext as ScrolledText
except ImportError:
    import Tkinter as tk # Python 2.x
    import ScrolledText

class datalogger():
    '''
    Generic Datalogger Class
    '''

    def __init__(self, datalog_path):
        #Initialize the class with Datalog file path
        self.max_file_size = 0
        #Check if datalog path is a file or a directory
        if not os.path.isdir(datalog_path):
            self.datalog_filepath = datalog_path #If datalog_path is file assign it to datalog_filepath variable
            self.datalog_dir = os.path.split(self.datalog_filepath)[0] #Extract datalog directory from datalog filepath
            self.check_if_folder_exists(self.datalog_dir) #Create datalog directory if it doesn't exists
        else:
            self.datalog_dir = datalog_path
            self.check_if_folder_exists(self.datalog_dir) #Create datalog directory if it doesn't exists
            self.datalog_filepath = os.path.join(self.datalog_dir,('log_'+self.get_time()+".csv"))
        print(self.get_time())
        self.init()
        return
    
    def check_if_folder_exists(self,folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return
    
    def get_file_size(self):
        #Get Datalog File size in Bytes
        return os.path.getsize(self.datalog_filepath)

    def get_time(self):
        #Returns Timestamp in MM-DD-YYYY-HH.MM format
        now = datetime.datetime.now()
        return str(now.strftime("%m-%d-%Y-%H.%M.%S"))
    
    def get_log(self, length):
        #Returns N lines from Datalog, where N is Specified by Variable 'length'
        line = self.datalog_fileref.readlines()
        return line
    
    def set_log(self, category, sub_category, log_string):
        #Logs the incoming entires (Category, Subcategory, String) with timestamp
        #Category = self.calss_name = self.__class__.__name__
        #Subcategory = inspect.getframeinfo(inspect.currentframe()).function
        timestamp = self.get_time()
        line = category + ',' + sub_category + ',' + log_string + "," + timestamp +'\n'
        self.open()
        self.datalog_fileref.writelines(line)
        self.close()
        logging.info(line)
        return
    
    def log_execution_time(self):
        #Returns the execution time on the module for logging
        return
    
    def worker(self):
        self.worker_root = tk.Tk()
        myGUI(self.worker_root)
        self.worker_root.mainloop()
        return
    
    def destroy_debugconsole(self):
        self.worker_root.destroy()
    
    def init(self):
        if (os.stat(self.datalog_filepath).st_size==0):
            line = 'category' + ',' + 'sub_category' + ',' + 'log_string' + "," + 'timestamp' +'\n'
            self.open()
            self.datalog_fileref.writelines(line)
            self.close()
        threading.Thread(target=self.worker, args=[]).start()
        return
    
    def open(self):
        self.datalog_fileref = open(self.datalog_filepath,'a+')
        return
    
    def close(self):
        self.datalog_fileref.close()
        return
    
    def show_logger(self):
        #Separate thread to display & use queue to refresh datalog
        logger_gui = tkinter_app.tkinter_app()
        window_title  = 'Datalogger'
        threading.Thread(target=logger_gui.progressbar_app, args=(window_title,)).start()
        return
    
class TextHandler(logging.Handler):
    # This class allows you to log to a Tkinter Text or ScrolledText widget
    # Adapted from Moshe Kaplan: https://gist.github.com/moshekaplan/c425f861de7bbf28ef06

    def __init__(self, text):
        # run the regular Handler __init__
        logging.Handler.__init__(self)
        # Store a reference to the Text it will log to
        self.text = text

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text.configure(state='normal')
            self.text.insert(tk.END, msg + '\n')
            self.text.configure(state='disabled')
            # Autoscroll to the bottom
            self.text.yview(tk.END)
        # This is necessary because we can't modify the Text from other threads
        self.text.after(0, append)

class myGUI(tk.Frame):

    # This class defines the graphical user interface 

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.build_gui()

    def build_gui(self):                    
        # Build GUI
        self.root.title('TEST')
        self.root.option_add('*tearOff', 'FALSE')
        self.grid(column=0, row=0, sticky='ew')
        self.grid_columnconfigure(0, weight=1, uniform='a')
        self.grid_columnconfigure(1, weight=1, uniform='a')
        self.grid_columnconfigure(2, weight=1, uniform='a')
        self.grid_columnconfigure(3, weight=1, uniform='a')
        #self.grid_columnconfigure(4, weight=1, uniform='a')
        #self.grid_columnconfigure(5, weight=1, uniform='a')
        

        # Add text widget to display logging info
        st = ScrolledText.ScrolledText(self, state='disabled', width=110)
        st.configure(font='TkFixedFont')
        st.grid(column=0, row=1, sticky='w', columnspan=4)

        # Create textLogger
        text_handler = TextHandler(st)

        # Logging configuration
        logging.basicConfig(filename='test.log',
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s')        

        # Add the handler to logger
        logger = logging.getLogger()        
        logger.addHandler(text_handler)
    
# from classes import datalogger
# import json_api
# 
# 
# json_filename = r"D:\Python\Tensorflow\Neural_Networks\Image Classifier\Generic_Classifier\main.json"
# json_obj = json_api.read_json(json_filename)
# datalogger_obj = datalogger.datalogger(json_obj['datalog_dir'])
# print(datalogger_obj.datalog_filepath)
# print(datalogger_obj.datalog_dir)
# datalogger_obj.set_log('category', 'sub_category', 'log_string')
# print("File size : "+ str(datalogger_obj.get_file_size())+' bytes')
# print(datalogger_obj.get_log(3))
        
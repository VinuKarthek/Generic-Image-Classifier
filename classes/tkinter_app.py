from tkinter import *
from tkinter import filedialog

from tkinter.ttk import *
import os, time


class tkinter_app():
    def __init__(self):
        #Define a Queue to exchange messages
        return
    
    def init(self, window_title = 'Generic GUI App', geometry = '350x200'):
        self.root = Tk()
        self.root.title(window_title)
        self.root.geometry(geometry)
        return    
    
    def quit(self):
        self.root.destroy()
    
    def progressbar_app(self, window_title):
        #This function displays Progressbar
        self.progress_value = 0
        self.progress_label_value = ''
        self.init(window_title, geometry = '300x40')
        self.folder_path = StringVar(self.root)
        self.folder_path.set("")
        progress_label = Label(self.root,textvariable=self.folder_path)
        progress_label.pack()
        progress=Progressbar(self.root,orient=HORIZONTAL,length=300,mode='determinate')
        progress.pack()
        #This loop runs till the operation hits 100%
        while self.progress_value < 100:
            self.folder_path.set(self.progress_label_value)
            progress['value'] = self.progress_value
            self.root.update()
        self.root.destroy()
        self.root.mainloop()     
        return      
    
    def debug_console_app(self, window_title):
        self.init(window_title, geometry = '300x40')
        self.log_console_data = StringVar(self.root)
        console_label = Label(self.root,textvariable=self.log_console_data)
        console_label.pack()
        self.root.protocol('WM_DELETE_WINDOW', self.quit)
        self.root.mainloop()
        return
    
    def browse_popup_app(self,window_title):
        self.init(window_title, geometry = '300x80')
        self.folder_path = StringVar(self.root)
        self.folder_path.set("Select Dataset folder or JSON Config file")
        self.label_foldername = Label(self.root,textvariable=self.folder_path).pack()
        button_browse = Button(text="Browse", command=self.browse_button)
        button_browse.pack()
        button = Button(self.root, text = 'Quit', command=self.quit)
        button.pack()
        self.root.mainloop()
        return self.folder_path.get()
    
    def browse_button(self):
        # Allow user to select a directory and store it in global var
        # called folder_path
        filename = filedialog.askdirectory()
        if(os.path.isdir(filename) or os.path.isfile(filename)):
            self.folder_path.set(filename)
            
    def __del__(self):
        return
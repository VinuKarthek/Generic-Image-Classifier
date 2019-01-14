'''
Created on Sep 20, 2018

@author: Vinu Karthek
'''

import paho.mqtt.client as mqtt
import inspect

class mqtt_api():
    ''' JSON Template --->
    "mqtt": {
            "host" : "localhost",
            "port" : 1883,
            "topics": {"sensors" : {"aliasname": "unique_topic_id", "tsense": "sensor/temperature"},
                    "actuators" : {"aliasname": "unique_topic_id", "tsense": "actuator/lever"}}
            }
    '''
    def __init__(self, json_object, datalogger_obj):
        '''
        Constructor
        '''
        self.load_settings_from_json(json_object)
        self.datalog_dir = ''
        self.mqttc = mqtt.Client()
        self.mqttc.connect(host = self.host, port = self.port)#iot.eclipse.org")
        self.mqttc.loop_start()
        self.set_logger(datalogger_obj)
        return
    
    def load_settings_from_json(self, json_object):
        #Read all settings from JSON file
        self.host = json_object['mqtt']['host']
        self.port = int(json_object['mqtt']['port'])
        self.topics = json_object['mqtt']['topics']         
        return
    
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

    def mqtt_publish(self,type, name , payload):
        
        if type in self.topics:
            if name in self.topics[type]:
                self.mqttc.publish(self.topics[type][name], payload = payload, qos = 1)
                print("Published")
            else:
                self.log("Error : '"+ name+ "' Not found in '"+ type+"'")
        else:
            self.log("Error : '"+ type+ "' Not found in 'Topics'")
        return
    
    
'''
from classes import mqtt_api
import json_api

json_obj = json_api.read_json(json_filename)
mqtt_obj = mqtt_api.mqtt_api(json_obj)
mqtt_obj.mqtt_publish(type = 'sensors', name = 'tsense', payload = "Hello")
'''
'''
Created on Sep 20, 2018

@author: Vinu Karthek
'''
import os, math, time, inspect, threading
import tensorflow as tf
import pandas as pd
import numpy as np
from classes import tkinter_app
import matplotlib.pyplot as plt
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data  #To Be Removed


class neurons(object):
    '''
    *No of layers - Name of each layers to display in Tensorboard
    *Type of the layer - Activation Function, RCNN/CNN/NN, No of Neurons
    *Dropout, Learning Rate, Training Algorithm
    *Accuracy & Entropy
    *Save Models
    '''


    def __init__(self, json_obj, dataset_obj, datalogger_obj = ''):
        '''
        Constructor
        '''
        self.load_settings_from_json(json_obj)
        self.biases = []
        self.weights = []
        self.layers = []
        self.metrics = []
        self.read_neural_schema()
        self.dataset_obj = dataset_obj
        self.x_res = dataset_obj.x_resolution
        self.y_res = dataset_obj.y_resolution
        self.image_channel = 1 # RGB or Greayscale or Binary Image
        self.labels = len(dataset_obj.label_names)
        #Datalogger
        self.set_logger(datalogger_obj)
        self.log(self.get_tf_version())
        self.log(os.path.split(self.neural_schema_path)[1])
    
    def get_tf_version(self):
        return tf.__version__    # @UndefinedVariable
    
    def toprint(self, printcontent):
        if(True):
            print ('{}'.format(printcontent))    
            
    def str2bool(self, string):
        if str(string) =='nan':
            return False
        else:
            return string.lower() in ("yes", "true", "1", "t")

    def strip_whitespaces(self,unformatted_string):
        #Removes/strips empty spaces from either side of strings in a list
        formatted_string = []
        for element_read in unformatted_string:
            element_read_stripped = element_read.strip()
            formatted_string.append(element_read_stripped)
        return formatted_string
    
    def check_if_folder_exists(self,folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return
    
    def open_progressbar(self):
        self.progress_gui = tkinter_app.tkinter_app()
        window_title  = 'Training Neural Network'
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
       
    def load_settings_from_json(self, json_obj):
        self.neural_schema_path = json_obj['neurons']['neural_schema']
        self.neural_schema_name = os.path.split(self.neural_schema_path)[1].strip('.csv')
        self.framework = json_obj['neurons']['framework']
        self.optimizer = json_obj['train']['optimizer']
        self.min_learning_rate = json_obj['train']['min_lr']
        self.max_learning_rate = json_obj['train']['max_lr']
        self.decay_speed = json_obj['train']['decay_speed']
        self.steps = int(json_obj['train']['steps'])
        self.update_interval = int(json_obj['train']['update_interval'])
        self.checkpoints = os.path.join(json_obj['checkpoints']['path'],self.neural_schema_name)
        self.check_if_folder_exists(self.checkpoints)
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
    
    def calculate_learning_rate(self, i=0):
        return (self.min_learning_rate + (self.max_learning_rate - self.min_learning_rate) * math.exp(-i/self.decay_speed))
    
    def shape(self,tensor):
        s = tensor.get_shape()
        return tuple([s[i].value for i in range(0, len(s))])
 
    def read_neural_schema(self):
        #Read and store nueral Schema in Pandas dataframe
        self.neural_schema_df = pd.read_csv(self.neural_schema_path, dtype = 'str')
        
        #Assign Default layer Name
        layers_list = self.neural_schema_df['Layer Name'].tolist()
        self.no_of_layers = len(layers_list)
        for i in range(self.no_of_layers):
            if str(layers_list[i]== 'NaN'):
                if i == 0:
                    layer_name = 'input_layer'
                elif i == (len(layers_list)-1):
                    layer_name = 'final_layer'
                else:
                    layer_name = 'layer'+str(i)
                self.neural_schema_df.set_value(i,'Layer Name', layer_name)
        
        #Print Neural Schema
        self.toprint(self.neural_schema_df)
        return
    
    def build_graph(self):
        # This function loops through every rows in the neural schema & initializes weights, biases, layers
        #and other parameters
        self.define_layers() #Loop through every row in neural schema
        self.define_metrics() #Metrics defines various metrics like accuracy,efficieny, entrophy etc
        self.log('Graph Build Complete')
        return        
    
    def init_weights(self):
        # Weights initialised with small random values between -0.2 and +0.2
        name = 'W_'+str(self.current_layer_index)
        is_convolution = self.str2bool(self.neural_schema_df['Convolution'][self.current_layer_index])
        is_log_summary = self.str2bool(self.neural_schema_df['Log_W'][self.current_layer_index])

        #Check if it is a convolutional layer
        if is_convolution:
            #If convolutional layer, read Filter, Input channel,output channel, stride
            output_channel = int(self.neural_schema_df['Channel'][self.current_layer_index])
            #Get X, Y Filter settings for 2D Convv
            filter = self.neural_schema_df['Filter'][self.current_layer_index]
            filter = filter.strip('').split(';')
            if len(filter) >1:
                filter_x = int(filter[0])
                filter_y = int(filter[1])
            else:
                filter_x = filter_y = int(filter[0])

            if(self.current_layer_index == 0):
                input_channel = self.image_channel
            else:
                input_channel = int(self.neural_schema_df['Channel'][self.current_layer_index-1])
            weight_object = tf.Variable(tf.truncated_normal([filter_x, filter_y, input_channel, output_channel], stddev=0.1), name = name)
        else:
            current_layer_neurons = [int(self.neural_schema_df['Neurons'][self.current_layer_index]), self.labels] [self.current_layer_index == self.no_of_layers-1]
            if(self.current_layer_index == 0):
                prev_layer_neurons = self.x_res*self.y_res
            else:
                prev_layer_index = self.current_layer_index-1
                is_prev_layer_convolution = self.str2bool(self.neural_schema_df['Convolution'][prev_layer_index])
                if is_prev_layer_convolution:
                    shape = self.shape(self.layers[prev_layer_index])
                    prev_layer_neurons = shape[1]
                else:
                    prev_layer_neurons = int(self.neural_schema_df['Neurons'][prev_layer_index])
            weight_object = tf.Variable(tf.truncated_normal([prev_layer_neurons, current_layer_neurons], stddev=0.1), name = name)
    
        #Insert Weight Object to array    
        self.weights.insert(self.current_layer_index, weight_object)
        if is_log_summary:
            self.variable_summaries(weight_object, name)
        self.log(str(weight_object))
        return weight_object
    
    def init_biases(self):
        #This function defines the biases for each layers as per settings from Neural Schema
        #Current Layer Neurons from NS, Label size is choosen for Last layer
        current_layer_neurons = [int(self.neural_schema_df['Neurons'][self.current_layer_index]), self.labels] [self.current_layer_index == self.no_of_layers-1]
        activation_function = self.neural_schema_df['Activation Function'][self.current_layer_index]
        name = 'B_'+str(self.current_layer_index)
        is_convolution = self.str2bool(self.neural_schema_df['Convolution'][self.current_layer_index]) #Convolutional Layer
        is_log_summary = self.str2bool(self.neural_schema_df['Log_B'][self.current_layer_index]) #Tensorboard summary
        
        #Check if it is a convolutional layer
        if is_convolution:
            input_channel = int(self.neural_schema_df['Channel'][self.current_layer_index])
            bias_object =  tf.Variable(tf.zeros([input_channel]), name = name)
        else:
            if(activation_function == 'relu'):
                # When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([neurons])/10
                bias_object =   tf.Variable(tf.ones([current_layer_neurons])/10, name = name)
            else:
                bias_object =  tf.Variable(tf.zeros([current_layer_neurons]), name = name)
        #Insert Bias Object to array 
        self.biases.insert(self.current_layer_index , bias_object)
        if is_log_summary:
            self.variable_summaries(bias_object, name)
        self.log(str(bias_object))
        return bias_object
    
    def define_layers(self):
        #This function define weights, biases, inputs, outputs of indivudual layers
        self.log('Initializing Graphs')
        self.Y_ = tf.placeholder(tf.float32, [None, self.labels], name = 'Y_') # Y_ contains all the correct answers
        is_convolution = self.str2bool(self.neural_schema_df['Convolution'][0]) #Check if this layer is convolutional
        
        with tf.name_scope('inputs'):
            # input X: x_res(x)y_res grayscale images, the first dimension (None) will index the images in the test-batch
            #Reshape X from 2D to 1D array if CNN/RNN is not used
            self.X = tf.placeholder(tf.float32, [None, self.x_res,self.y_res, self.image_channel], name = 'X')
            self.X = [tf.reshape(self.X, [-1, self.x_res*self.y_res]), self.X][is_convolution]
        Y_in =  self.X   
        
        #Loop through every settingg & create respective layers
        for layer_index in range(self.no_of_layers):
            self.current_layer_index = layer_index
            name = 'L_'+str(self.current_layer_index)
            is_convolution = self.str2bool(self.neural_schema_df['Convolution'][self.current_layer_index]) #Check if this layer is convolutional
            layer_name = self.neural_schema_df['Layer Name'][self.current_layer_index]
            activation_function = self.neural_schema_df['Activation Function'][self.current_layer_index]
            pkeep = self.neural_schema_df['pkeep'][self.current_layer_index]
            if not str(pkeep) =='nan':
                pkeep = float(pkeep)
            else:
                pkeep = 1
            self.log('\n'+layer_name)
            if(is_convolution):
                stride = int(self.neural_schema_df['Stride'][self.current_layer_index])
                Y_temp = tf.nn.conv2d(Y_in, self.init_weights(), strides=[1, stride, stride, 1], padding='SAME') + self.init_biases()
                #Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1, convolutional=True)
                is_next_layer_convolution = self.str2bool(self.neural_schema_df['Convolution'][self.current_layer_index+1])
                is_reshape = not is_next_layer_convolution
            else: 
                Y_temp = tf.matmul(Y_in, self.init_weights()) + self.init_biases()
                is_reshape = False  
            #Define layer       
            with tf.name_scope(layer_name):
                #Define Y_OUT as per the activation finctions, more functions to be added
                if activation_function == 'relu' : 
                    Y_out = tf.nn.relu(Y_temp, name = name)
                elif activation_function == 'sigmoid':
                    Y_out = tf.nn.sigmoid(Y_temp, name = name)
                elif activation_function == 'tanh':
                    Y_out = tf.nn.tanh(Y_temp, name = name)
                else: #Final Layer must always be Softmax
                    Y_out = tf.nn.softmax(Y_temp, name = name)
                #Reshape Y_Out from 2D to 1D if the next layer is not 2D Convolutional
                if is_reshape:
                    #next_layer_neurons =int(self.neural_schema_df['Neurons'][self.current_layer_index+1]) 
                    shape_Y_Out = self.shape(Y_out)
                    Y_out = tf.reshape(Y_out, shape=[-1, shape_Y_Out[1] * shape_Y_Out[2] * shape_Y_Out[3]])
                #Define dropout    
                dropout = self.str2bool(self.neural_schema_df['Dropout'][self.current_layer_index])                    
                Y_out = [Y_out, tf.nn.dropout(Y_out, pkeep)][dropout]
            #Store the created layer into the layers array for future use
            self.layers.insert(self.current_layer_index, Y_out)
            self.log(str(Y_out))
            #Connect the output of current layer to the input of the next layer
            Y_in = Y_out
                
        self.Y = self.layers[self.no_of_layers-1] 
        self.log('Layer Initialization Complete')
        return
    
    def define_metrics(self):
        self.log('Initializing Metrics')
        with tf.name_scope('metrics'):
            self.learning_rate = tf.placeholder(tf.float32)
            #Define cross entropy
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.Y, labels=self.Y_, name = 'cross_entropy')
            self.cross_entropy = tf.reduce_mean(self.cross_entropy)*100
            tf.summary.scalar('cross_entropy', self.cross_entropy)
            # accuracy of the trained model, between 0 (worst) and 1 (best)
            self.correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        
        # matplotlib visualisation
        #allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1])], 0)
        #allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1])], 0)
        # to use for sigmoid
        #allactivations = tf.concat([tf.reshape(Y1, [-1]), tf.reshape(Y2, [-1]), tf.reshape(Y3, [-1]), tf.reshape(Y4, [-1])], 0)
        # to use for RELU
        #allactivations = tf.concat([tf.reduce_max(Y1, [0]), tf.reduce_max(Y2, [0]), tf.reduce_max(Y3, [0]), tf.reduce_max(Y4, [0])], 0)
        
        with tf.name_scope('train_step'):
            if (self.optimizer == 'AdamOptimizer'):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif (self.optimizer == 'RMSProp'):
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif (self.optimizer == 'ProximalGDO'):
                optimizer = tf.train.ProximalGradientDescentOptimizer(self.learning_rate)
            else:
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            #Initialize the minimization function with Optimizer & Entropy functions    
            self.train_step = optimizer.minimize(self.cross_entropy)
        self.log('Metric Initialization Complete')
        return
    
    def init_session(self):
        self.init = tf.global_variables_initializer()
        session = tf.Session() #Initialize Session
        return session
    
    def execute_session(self):
        self.log('Executing Session...')
        #Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.checkpoints)     
        self.log('Writers Initialized Successfully')
        #Initialize Session & Variables
        sess = self.init_session()
        sess.run(self.init) # Initialize Global variables
        self.log('Session Initialized Successfully')
        self.log('Training Start...')
        #Training Loop
        self.open_progressbar()
        for i in range (self.steps):
            start_time = time.time()
            #
            if i % self.update_interval == 0: # Record summaries and test-set accuracy
                batch_X, batch_Y = self.dataset_obj.next_test_batch(100)
                summary, acc, ent = sess.run([merged, self.accuracy, self.cross_entropy], feed_dict= {'inputs/X:0' : batch_X, self.Y_:batch_Y})
                self.log(str(i) + ": accuracy:" + str(acc*100) + " loss: " + str(ent) + " (lr:" + str(self.calculate_learning_rate(i)) + ")")
                self.writer.add_summary(summary, i)
            else:
                batch_X, batch_Y = self.dataset_obj.next_train_batch(100)
                sess.run(self.train_step, feed_dict = {'inputs/X:0' :batch_X, self.Y_:batch_Y, self.learning_rate: self.calculate_learning_rate(i)})
            self.update_progressbar(i,self.steps,start_time)
        self.close_progressbar()
        self.log('Training Complete')
        self.save_session(sess)
        return
        
    def save_session(self,sess):
        #This function saves the tensroflow graph, summaries & cehckpoints for future use
        #Writer for Tensorboard Graphs & Summaries
        self.writer.add_graph(sess.graph)
        #Saver for saving Checkpoints
        save_path = os.path.join(self.checkpoints,self.neural_schema_name+".ckpt")
        self.saver = tf.train.Saver() #Initialize Saver 
        self.saver.save(sess, save_path)
        #Save Grapth as Text file
        self.save_graph_as_txt()
        self.log('Session Saved Successfully')
        return
    
    def save_graph_as_txt(self):
        #Saves graph as readable txt file
        graph_txt_path = os.path.join(self.checkpoints, self.neural_schema_name +'_graph.txt')
        with open(graph_txt_path, 'w') as f:
            f.write(str(tf.get_default_graph().as_graph_def()))
        return
    
    def load_session(self):
        sess = self.init_session()
        self.saver = tf.train.Saver() #Initialize Saver 
        load_path = os.path.join(self.checkpoints,self.neural_schema_name+".ckpt")
        self.saver.restore(sess, load_path)
        
        batch = self.dataset_obj.next_test_batch(1)
        result = sess.run(self.layers[self.no_of_layers-1], feed_dict = {'inputs/X:0':batch[0]})#,Y_:mnist.test.labels})
        
    
        plotData = batch[0]
        plotData = plotData.reshape(28, 28)
        plt.gray() # use this line if you don't want to see it in color
        plt.title("The Prediction is:"+ str(np.argmax(result)))
        plt.imshow(plotData)
        plt.show()
                
        self.log('Session Loaded Successfully')
        return
    
    def variable_summaries(self,var, summary_name = 'summary'):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(summary_name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
    
    def embed_images_for_visualization(self):
        #Embed Images to Tensorboard for Visualization (t-SNE, PSE ect)
        #Store all the test images from Dataset object into a Tensor Variable
        #mnist = input_data.read_data_sets(os.getcwd() + "/data/", one_hot=True)
        reshaped = np.resize(self.dataset_obj.data_dict['test_X'], [len(self.dataset_obj.data_dict['test_Y']), self.x_res*self.y_res])
        images = tf.Variable(reshaped)
        print(self.shape(images))

        #Open Metadata TSV File & write all the labels to it
        metadata = os.path.join(self.checkpoints, 'metadata.tsv')
        with open(metadata, 'w') as metadata_file:
            for row in range(len(self.dataset_obj.data_dict['test_Y'])):
                c = np.nonzero(self.dataset_obj.data_dict['test_Y'][::1])[1:][0][row]
                metadata_file.write('{}\n'.format(c))
        #Initialize & Save session to save the Variable
        with tf.Session() as sess:
            saver = tf.train.Saver([images])

            sess.run(images.initializer)
            saver.save(sess, os.path.join(self.checkpoints,self.neural_schema_name+".ckpt"))
            print(os.path.join(self.checkpoints,self.neural_schema_name+".ckpt"))
            config = projector.ProjectorConfig()
            # One can add multiple embeddings.
            embedding = config.embeddings.add()
            embedding.tensor_name = images.name
            # Link this tensor to its metadata file (e.g. labels).
            embedding.metadata_path = metadata
            # Saves a config file that TensorBoard will read during startup.
            projector.visualize_embeddings(tf.summary.FileWriter(self.checkpoints), config)
        self.log('Test Images Embedded for Tensorboard Visualization')
        return
        
# from classes import neurons
# import json_api
# 
# 
# json_filename = r"D:\Python\Tensorflow\Neural_Networks\Image Classifier\Generic_Classifier\main.json"
# dataset_obj = ''
# json_obj = json_api.read_json(json_filename)
# neuron_obj = neurons.neurons(json_obj,dataset_obj)
# print (neuron_obj.get_tf_version())
# neuron_obj.build_graph()
# neuron_obj.execute_session()
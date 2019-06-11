# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import csv
import numpy as np
import openpyxl
import tensorflow as tf
# Clear cache before running
tf.reset_default_graph()
# Set some parameters
Lrate = 0.001 # Learning rate
Epoch = 20 # Total training epoch
Batch = 20 # Batch size, Iteration = Data size//Batch
HidenNum = 5 # Number of hiden layer neurons
F_n = 10 # Fold Count

# Images loading
def load_images_labels_RGB(img_folder, label_file):
    with open(label_file) as f:
        reader = csv.reader(f)
        next(reader) # skip header
        labels = []
        images = []
        for row in reader:
            img_filename = row[0]
            img = Image.open(os.path.join(img_folder,img_filename))
            if img is not None:
                images.append(np.array(img))
                labels.append(int(row[1]))
    return np.array(images), np.array(labels)

# Import data (normalized)
Img_Data, Label_Data = load_images_labels_RGB('train/', 'train.csv')
Size_Data = Img_Data.shape
Fold_size = Size_Data[0]//F_n
Mean_RGB = np.array([128.41563722, 115.24518493, 119.38645491])
Std_RGB = np.array([38.55379149, 35.64913446, 39.07419321])
Data_Norm = (Img_Data - Mean_RGB)/Std_RGB

# Build BPNN
Data_in = tf.placeholder(dtype= tf.float32, shape=(None, 32, 32, 3), name="inputs")
# Note that the ground truth is one-hot format
Data_y = tf.placeholder(dtype= tf.int32, shape=(None), name="GroundTue")
#depth = 7
Data_y_ = tf.one_hot(Data_y, 2)
#TrainFlg = tf.placeholder(dtype= tf.bool)
learningRate = tf.placeholder(tf.float32)

#Flat_in = tf.reshape(Data_in, [-1, 32*32*3])
Cnn = tf.layers.conv2d(Data_in, 8, [3, 3], strides=(1, 1), padding="SAME", activation=tf.nn.relu)
Cnn = tf.layers.conv2d(Cnn, 6, [3, 3], strides=(1, 1), padding="SAME", activation=tf.nn.relu)
#Cnn = tf.nn.max_pool(Cnn, ksize=[1, 2, 2, 1],strides=[1, 1, 1, 1], padding='VALID')
Cnn = tf.reshape(Cnn, [-1, 32*32*6])
Cnn = tf.layers.dense(Cnn, 32, activation=tf.nn.sigmoid, trainable=True, name='Input_L')
Cnn = tf.layers.dense(Cnn, 2, activation=tf.nn.sigmoid, trainable=True, name='Output_L')
Out_y = tf.identity(Cnn, name = "prediction")
# MSE error
loss = tf.sqrt(tf.reduce_mean(tf.square(Data_y_ - Out_y)), name="loss")
# Set Adam Optimizer
train_step = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss, name="train_step")
correct_prediction = tf.equal(tf.argmax(Out_y, 1), tf.argmax(Data_y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Start training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    per = np.random.permutation(Data_Norm.shape[0])
    Shuf_Data_Norm = Data_Norm[per, :, :, :]
    Shuf_Label_Data = Label_Data[per]
    for Ep_CNT in range(Epoch):
        Overall_acc = 0.0
        Overall_loss = 0.0
        for i in range(F_n-1):
            DataN_te = Shuf_Data_Norm[Fold_size*i:Fold_size*(i+1), :, :, :]
            DataN_tr_1 = Shuf_Data_Norm[0:(Fold_size*i), :, :, :]
            DataN_tr_2 = Shuf_Data_Norm[Fold_size*(i+1):, :, :, :]
            DataN_tr = np.concatenate((DataN_tr_1,DataN_tr_2))
            DataN_te_y = Shuf_Label_Data[Fold_size*i:Fold_size*(i+1)]
            DataN_tr_y_1 = Shuf_Label_Data[0:(Fold_size*i)]
            DataN_tr_y_2 = Shuf_Label_Data[Fold_size*(i+1):]
            DataN_tr_y = np.concatenate((DataN_tr_y_1,DataN_tr_y_2))
            
            Iteration = DataN_tr.shape[0]//Batch
            for it_CNT in range(Iteration):
                Data_tr_x = DataN_tr[it_CNT*Batch:(it_CNT+1)*Batch, :, :, :]
                Data_tr_y = DataN_tr_y[it_CNT*Batch:(it_CNT+1)*Batch].astype(int)
                #depth = 7 # Depth after one-hot conversion
                #Data_tr_y_onehot = tf.one_hot(Data_tr_y, depth)
                _, batch_loss = sess.run([train_step, loss],feed_dict={Data_in: Data_tr_x, Data_y: Data_tr_y, 
                                         learningRate:Lrate})
                batch_acc = sess.run([accuracy],feed_dict={Data_in: Data_tr_x, Data_y: Data_tr_y, 
                                     learningRate:Lrate})
                '''
                print("iteration: %d, batch_loss: %.4f, batch_acc: %.4f"
                          % (it_CNT, batch_loss, batch_acc[0]))
                '''
            Data_va_x = DataN_te[0:100, :, :, :]
            Data_va_y = DataN_te_y[0:100].astype(int)
            Val_loss, Val_acc = sess.run([loss, accuracy],feed_dict={Data_in: Data_va_x, Data_y: Data_va_y, 
                                learningRate:Lrate})
            Overall_acc = Overall_acc + Val_acc
            Overall_loss = Overall_loss + Val_loss
            print("Fold: %d, batch_loss: %.4f, batch_acc: %.4f"
                  % (i, Val_loss, Val_acc))
        print("Epoch:    %d,  Overall_Accuracy:  %.4f, Overall_loss:  %.4f"
              % ((Ep_CNT+1), Overall_acc/(F_n-1), Overall_loss/(F_n-1)))
        DataN_te = Shuf_Data_Norm[Fold_size*9:, :, :, :]
        DataN_te_y = Shuf_Label_Data[Fold_size*9:]
        Data_te_x = DataN_te[0:100, :, :, :]
        Data_te_y = DataN_te_y[0:100].astype(int)
        Te_loss, Te_acc = sess.run([loss, accuracy],feed_dict={Data_in: Data_te_x, Data_y: Data_te_y, 
                                learningRate:Lrate})
        print("Epoch:   %d, Test_loss: %.4f, Test_acc: %.4f"
              % ((Ep_CNT+1), Te_loss, Te_acc))

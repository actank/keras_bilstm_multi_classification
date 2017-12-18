#coding:utf-8
import sys
from sklearn import datasets
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.externals import joblib

import numpy as np
import keras
from keras.models import Sequential,Model
from keras.preprocessing import sequence
from keras.layers import Embedding,Dense,Dropout,Activation,Flatten,LSTM,Reshape,ConvLSTM2D,SimpleRNN,MaxPooling1D,merge,Input,TimeDistributed
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.constraints import nonneg
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam,Adagrad,RMSprop
from keras import backend as K


def train():
    data = datasets.load_svmlight_file("./train.data")
    X = data[0]
    Y = data[1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
    model = LinearSVC(random_state=0)
    model.fit(X_train, Y_train)
    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    #对每个类别分别算pr
    #for c in (1,2,3,4):
    #c_index = []
    #for i in range(len(Y_train)):
    #    if Y_train[i] == c:
    #        c_index.append(i)
    #tmp_y_train = Y_train[[c_index]]
    #tmp_train_pred = train_pred[[c_index]]
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(Y_train,train_pred, labels=[1,2,3,4])
    for i in range(4):
        print("svm train pr:%f %f" % (precision[i], recall[i]))
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(Y_test, test_pred, labels=[1,2,3,4])
    for i in range(4):
        print("svm test pr:%f %f" % (precision[i], recall[i]))

    lr_model = LogisticRegression(penalty='l2',
                                  tol=0.0001, 
                                  C=0.01,
                                  solver='lbfgs',
                                  class_weight={1:8.0,2:5.0,3:3.0,4:1.0},
                                  #class_weight='balanced',
                                  multi_class='multinomial')
    lr_model.fit(X_train, Y_train)
    test_pred = lr_model.predict(X_test)
    train_pred = lr_model.predict(X_train)
    #对每个类别分别算pr
    #for c in (1,2,3,4):
    #c_index = []
    #for i in range(len(Y_train)):
    #    if Y_train[i] == c:
    #        c_index.append(i)
    #tmp_y_train = Y_train[[c_index]]
    #tmp_train_pred = train_pred[[c_index]]
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(Y_train,train_pred, labels=[1,2,3,4])
    for i in range(4):
        print("lr train pr:%f %f" % (precision[i], recall[i]))
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(Y_test, test_pred, labels=[1,2,3,4])
    for i in range(4):
        print("lr test pr:%f %f" % (precision[i], recall[i]))

  #为了避免划分train和test带来的特征损失，导致过不了sklearn的特征数检查这个坑爹设计，合并后训练模型，还要加上n_features这个选项，因为特征shape被csr指定了，又是个大坑
    lr_model.fit(X, Y)
    joblib.dump(lr_model, "lr.model")

    all_pred = lr_model.predict(X)
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(Y,all_pred, labels=[1,2,3,4])
    for i in range(4):
        print("all pr:%f %f" % (precision[i], recall[i]))


    submit_data = datasets.load_svmlight_file("./submit.data", n_features=X.shape[1])
    X_submit = submit_data[0]

    submit_pred = lr_model.predict(X_submit)
    with open("submit_data_index") as f, open("submit_result", "w") as f1:
        for pred in submit_pred:
            if pred == 1.0:
                pred = '春'
            elif pred == 2.0:
                pred = '夏'
            elif pred == 3.0:
                pred = '秋'
            else:
                pred = '冬'
            f1.write(str(pred) + " " + f.readline())

    return
def train_lstm():
    #data = datasets.load_svmlight_file("./train_lstm.data")
    #X = data[0]
    #Y = data[1]
    data = np.genfromtxt('train_lstm.data', delimiter=',', dtype=np.int32)
    X = data[:,1:]
    Y = data[:,0]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
    #对每个类别分别算pr
    #for c in (1,2,3,4):
    #c_index = []
    #for i in range(len(Y_train)):
    #    if Y_train[i] == c:
    #        c_index.append(i)
    #tmp_y_train = Y_train[[c_index]]
    #tmp_train_pred = train_pred[[c_index]]
    #model = Sequential()
    #model.add(Embedding(input_dim=int(np.max(X_train))+1,output_dim=10,input_length=X_train.shape[1]))
    #model.add(LSTM(6,return_sequences=False))
    #model.add(MaxPooling1D(pool_size=2, strides=None))
    #model.add(ConvLSTM2D(5, kernel_size=2, strides=1))
    #model.add(Flatten())
    #model.add(SimpleRNN(5,return_sequences=False))

    #model.add(Dense(4))
    #model.add(Activation('softmax'))
    #mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #-----------------------------------
    #model = Sequential()
    #model.add(Embedding(int(np.max(X_train))+1, 100, input_length=X_train.shape[1]))
    #model.add(LSTM(10, dropout=0.3, recurrent_dropout=0.2, return_sequences=True))
    #model.add(LSTM(10, dropout=0.3, recurrent_dropout=0.2, go_backwards=True))
    #model.add(Dense(4, activation='softmax'))
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['mae'])
    #-----------------------------------
    Y_train = Y_train.reshape(Y_train.shape[0],1).tolist()
    Y_test = Y_test.reshape(Y_test.shape[0],1).tolist()

    X_train = keras.preprocessing.sequence.pad_sequences(X_train, value=0.)
    X_test = keras.preprocessing.sequence.pad_sequences(X_test, value=0.)
    Y_train = keras.preprocessing.sequence.pad_sequences(Y_train, value=0.)
    Y_test = keras.preprocessing.sequence.pad_sequences(Y_test, value=0.)
    #指定dim不指定nsamples，但是python不能写(,dim)的形式，只能写(dim,)的形式
    sequence = Input(shape=(X_train.shape[1],), dtype='int32')

    embeded = Embedding(int(np.max(X_train))+1, 100, input_length=X_train.shape[1])(sequence)
    forwards = LSTM(10, dropout=0.3, recurrent_dropout=0.2)(embeded)
    backwards = LSTM(10, dropout=0.3, recurrent_dropout=0.2,go_backwards=True)(embeded)
    merged = concatenate([forwards, backwards], axis=-1)
    after_dp = Dropout(0.2)(merged)
    output = Dense(4, activation='softmax')(after_dp)
    model = Model(input=sequence, output=output)
    ##bug 在这里，loss函数必须是sparse_categorical_crossentropy？
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['mae'])

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mae'])
    #model.compile(loss='mean_squared_error', optimizer=mypotim, metrics=["mae"])
    model.fit(X_train, Y_train, batch_size=100, epochs=60, shuffle=True, verbose=True,validation_split=0.1)
    #inp = model.input
    #outputs = [model.layers[0]]
    #functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]
    #layer_outs = [func([X_train, 1.]) for func in functors]
    #print(layer_outs)
    #get_3rd_layer_output = K.function([model.layers[0].input],
    #                                  [model.layers[0].output])
    #layer_output = get_3rd_layer_output([X])[0][0]
    #print(layer_output)

    score = model.evaluate(X_test, Y_test, batch_size=100)
    print("mae:%f" % score[1])
    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    print(train_pred)
    train_pred = train_pred.tolist()
    train_pred_label = []
    for l in train_pred:
        mm = 0
        la = 0
        for j in range(4):
            if l[j] > mm:
                la = j
                mm = l[j]
        train_pred_label.append(la)
    test_pred = test_pred.tolist()
    test_pred_label = []
    for l in test_pred:
        mm = 0
        la = 0
        for j in range(4):
            if l[j] > mm:
                la = j
                mm = l[j]
        test_pred_label.append(la)


    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(Y_train,train_pred_label, labels=[0,1,2,3])
    for i in range(4):
        print("lstm train pr:%f %f" % (precision[i], recall[i]))
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(Y_test, test_pred_label, labels=[0,1,2,3])
    for i in range(4):
        print("lstm test pr:%f %f" % (precision[i], recall[i]))
    
    submit_data = np.genfromtxt('submit_lstm.data', delimiter=',', dtype=np.int32)
    X_submit = submit_data
    submit_pred = model.predict(X_submit)
    submit_pred = submit_pred.tolist()
    with open("submit_data_index_lstm") as f, open("submit_result_lstm", "w") as f1:
        for l in submit_pred:
            mm = 0
            la = 0
            for j in range(4):
                if l[j] > mm:
                    la = j
                    mm = l[j]

            if la == 0:
                pred = '春'
            elif la == 1:
                pred = '夏'
            elif la == 2:
                pred = '秋'
            else:
                pred = '冬'
            f1.write(str(pred) + " " + f.readline())



    return

if __name__ == "__main__":
    #train()
    train_lstm()

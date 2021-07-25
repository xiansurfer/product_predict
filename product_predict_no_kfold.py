import pandas as pd
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import LSTM,BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv1D,LSTM,Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
def mat_concat(id_list,df_data,df_target):
    data_list = []
    target_list = []
    for id in id_list:
        data_list.append(df_data[df_data['id']==id].drop(columns=['id']).values)
        target_list.append(df_target[df_target['id']==id].drop(columns=['id']).values)
    data_arr = np.vstack(data_list)
    target_arr = np.vstack(target_list)

    return data_arr,target_arr

def data_norm(train,validation):
    maximum = train.max(axis=0)
    minimum = train.min(axis=0)
    range = maximum - minimum
    train_norm = (train-np.tile(minimum,(train.shape[0],1)))/(np.tile(range,(train.shape[0],1)))
    validation_norm = (validation-np.tile(minimum,(validation.shape[0],1)))/(np.tile(range,(validation.shape[0],1)))
    return train_norm,validation_norm

def data_process():
    seed = random.seed(7)
    df_data = pd.read_csv('7.21故障井洗掉异常allwell.csv')
    df_target = df_data[['id','OIL(bbl)']]
    df_data.drop(columns=['RUL', 'GAS(103Mscf)',
           'WATER(bbl)', 'LIQUID(bbl)', 'GOR(Mscf/bbl)'],inplace=True)
    id_base = list(set(df_data['id'].values))

    #随机挑选20口井做验证
    validation_id = random.sample(id_base,20)

    #其余的井id放入train_id
    train_id = [id for id in id_base if id not in validation_id]

    train_data,train_target = mat_concat(train_id,df_data,df_target)
    validation_data,validation_target = mat_concat(validation_id,df_data,df_target)
    train_norm,validation_norm = data_norm(train_data,validation_data)


    train_window = [train_norm[i:i+7] for i in range(train_norm.shape[0]-7-1)]
    target_window = [train_target[i+7] for i in range(train_norm.shape[0]-7-1)]
    train_window = np.array(train_window)
    target_window = np.array(target_window)

    validation_window = [validation_norm[i:i+7] for i in range(validation_norm.shape[0]-7-1)]
    validation_target_window = [validation_target[i+7] for i in range(validation_norm.shape[0]-7-1)]
    validation_window = np.array(validation_window)
    validation_target_window = np.array(validation_target_window)

    train_window = train_window.reshape(train_window.shape)
    return train_window,target_window,validation_window,validation_target_window

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=200, input_shape=(input_shape), return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.3))
    model.add(LSTM(units=150))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(Dense(1))
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
    model.compile(loss='mae', optimizer=adam)
    model.summary()

    return model


x_train,y_train,x_validation,y_validation = data_process()
model = build_model(x_train.shape[1:])
filepath = 'weights.best_no_kfold.h5'
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',verbose=1,
                             save_best_only=True,mode='min')
callbacklist = [checkpoint]
history = model.fit(x_train,y_train,validation_data=(x_validation,y_validation),
          batch_size=300,epochs=50,verbose=2,callbacks=callbacklist)
model.save('weights.best_no_kfold.h5')
val_loss = history.history['val_loss']
loss = history.history['val_loss']
time = [x for x in range(len(loss))]

plt.grid()
plt.plot(time,loss)
plt.plot(time,val_loss)
plt.show()
plt.savefig('loss.png')




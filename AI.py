import numpy as np
# mlp for multi-label classification
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

X_test =  np.random.rand(64,1024)
X_train = np.random.rand(1024,1024)

Y_test = np.random.rand(64,32)
Y_train = np.random.rand(1024,32)


def get_model_MLP(n_inputs, n_outputs):
	model = Sequential()
	#model.add(Dense(512, input_dim=n_inputs, kernel_initializer='random_normal', activation='relu'))
	#model.add(Dropout(0.25))
	model.add(Dense(256, input_dim=n_inputs, kernel_initializer='random_normal', activation='relu'))
	model.add(Dropout(0.25))
	#model.add(Dense(128, input_dim=256, kernel_initializer='random_normal', activation='relu'))
	#model.add(Dropout(0.25))
	model.add(Dense(64, input_dim=256, kernel_initializer='random_normal', activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(n_outputs,input_dim=64, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

def train_model(X, y):
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    callback = EarlyStopping(monitor='loss', patience=10)
    model = get_model_MLP(n_inputs,n_outputs)
    model.fit(X_train, y, verbose=1, epochs=1000,callbacks = [callback])
    return model

def test_model(model,X_test,y_test):
    print(X_test)
    results = model.predict(X_test)
    return results


model = train_model(X_train,Y_train)
results = test_model(model,X_test,Y_test)
print(results)
pickle.dump(model,open("model.sav",'wb'))


model2 = pickle.load(open("model.sav",'rb'))
results = test_model(model2,X_test,Y_test)
print(results)
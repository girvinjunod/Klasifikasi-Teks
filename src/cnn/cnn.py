import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing.text import Tokenizer
# from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence
from sklearn.metrics import f1_score

path = "../../data_worthcheck/"
dataTrain = pd.read_csv(path + "train.csv")
dataTest = pd.read_csv(path + "test.csv")


X_train = dataTrain['text_a'].values
y_train = dataTrain['label']

X_test = dataTest['text_a'].values
y_test = dataTest['label']

maps = {"no": 0, "yes": 1}
y_train = y_train.replace(maps)
y_test = y_test.replace(maps)



tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
Xcnn_train = tokenizer.texts_to_sequences(X_train)
Xcnn_test = tokenizer.texts_to_sequences(X_test)
vocab_size = len(tokenizer.word_index) + 1  
# print(X_train[1])
# print(Xcnn_train[1])

# Padding the data samples to a maximum review length in words
max_words = 450
Xcnn_train = tf.keras.utils.pad_sequences(Xcnn_train, maxlen=max_words,  padding='post')
Xcnn_test = tf.keras.utils.pad_sequences(Xcnn_test, maxlen=max_words,  padding='post')
# Building the CNN Model
model = Sequential()      # initilaizing Sequential for CNN 
# Adding the embedding layer which will take in maximum of 450 words as input and provide a 32 dimensional output 
model.add(tf.keras.layers.Embedding(vocab_size, 32, input_length=max_words))
# Conv Layer
model.add(Conv1D(32, 3, padding='same', activation='relu'))
# Pooling Layer
model.add(MaxPooling1D())
model.add(Flatten())
# Dense Layer
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Optimize with SGD
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), metrics=['accuracy'])
model.summary()


# Fitting the model and search f1 score

model.fit(Xcnn_train, y_train,
                    epochs=3,
                    verbose=False,
                    validation_data=(Xcnn_test, y_test),
                    batch_size=16)
loss, accuracy = model.evaluate(Xcnn_train, y_train, verbose=False)
print("Training Accuracy: {:.6f}".format(accuracy))
loss, accuracy = model.evaluate(Xcnn_test, y_test, verbose=False)
print("Testing Accuracy:  {:.6f}".format(accuracy)) 


yhat_probs = model.predict(Xcnn_test, verbose=0)
predict_y = model.predict(Xcnn_test, verbose=0)
yhat_classes=np.argmax(predict_y,axis=1)
yhat_probs = yhat_probs[:, 0]
 

f1 = f1_score(y_test, yhat_classes, average='micro')
print('F1 score: %f' % f1)
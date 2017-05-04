import numpy as np
import sys
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

from keras.models import load_model
#K.set_image_dim_ordering('th')
num_classes = 7

def load_data():
	origin_data = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype='str')
	label = origin_data[:, 0].astype(int)

	origin_data = origin_data[:, 1]
	data = np.empty( (origin_data.shape[0], 48, 48, 1))
	for i in range(origin_data.shape[0]):
		tmp = np.asarray(origin_data[i].split(' '), dtype=np.float32)
		data[i,:, :,:]= tmp.reshape((48, 48, 1))
	return data, label
def load_test():
	test = np.genfromtxt(sys.argv[2], delimiter=',', skip_header=1, dtype='str')
	test = test[:,1]

	testdata = np.empty( (test.shape[0], 48, 48, 1))
	for i in range(test.shape[0]):
		tmp = np.asarray(test[i].split(' '), dtype=np.float32)
		testdata[i,:, :,:]= tmp.reshape((48, 48, 1))
	return testdata

np.set_printoptions(threshold='nan', suppress=True, precision=8)
data, label = load_data()
label = np_utils.to_categorical(label, num_classes)
data = data/255

val = data[:3000]
vallabel = label[:3000]
#training_data = (training_data - np.mean(training_data,axis = 0))/np.std(training_data,axis = 0)

model = Sequential()
model.add(Convolution2D(32,3,3,input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(Convolution2D(32,3,3))

model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(128,3,3))
model.add(Activation('relu'))
model.add(Convolution2D(128,3,3))

model.add(Activation('relu'))

# Fully connected part
model.add(Flatten())
model.add(Dense(output_dim=1000))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
epochs = 30
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
model.summary()
# Fit the model
datagen = ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)

datagen.fit(data[3000:])

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(data[3000:], label[3000:], batch_size=128),validation_data = (val, vallabel),steps_per_epoch=len(val)/10, epochs=epochs)


#model.fit(data,label,batch_size=250,epochs=epochs, shuffle=True, verbose=1, validation_split=0.2 )
# Final evaluation of the model
#scores = model.evaluate(data, label, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))

model.save('model.h5')

emotion_classifier = load_model('model.h5')
emotion_classifier.summary()

test_gogo = load_test()
test_gogo = test_gogo/255
ans = emotion_classifier.predict_classes(test_gogo,batch_size=128)
with open('Answer_1.csv','w') as f:
		f.write('id,label\n')
		for idx,a in enumerate(ans):
			f.write('{},{}\n'.format(idx,a))

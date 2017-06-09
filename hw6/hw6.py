from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import sys
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adadelta, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import genfromtxt
from keras.models import load_model
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers import Dot,Add
import keras
def get_model(n_users, n_items, latent_dim = 5):
	user_input = Input(shape=[1])
	item_input = Input(shape=[1])
	user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
	user_vec = Flatten()(user_vec)
	item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
	item_vec = Flatten()(item_vec)
	user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
	user_bias = Flatten()(user_bias)
	item_bias = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input)
	item_bias = Flatten()(item_bias)
	r_hat = Dot(axes=1)([user_vec, item_vec])
	r_hat = Add()([r_hat, user_bias, item_bias])
	model = keras.models.Model([user_input, item_input], r_hat)
	opt = Adam(lr = 0.001)
	model.compile(loss='mse', optimizer=opt)
	return model

def main():
	train_data = genfromtxt(sys.argv[1], delimiter=',')
	n_users = int(np.max(train_data[1:,1]) + 1)
	user = np.reshape(train_data[1:,1],(-1,1))
	n_items = int(np.max(train_data[1:,2]) + 1)
	item = train_data[1:,2].reshape((-1,1))
	rating = train_data[1:,3].reshape((-1,1))
	r_std = np.std(rating,axis = 0)
	r_mean = np.mean(rating,axis = 0)
	rating = (rating - np.mean(rating,axis = 0))/np.std(rating,axis = 0)
	model = get_model(n_users, n_items)
	model.fit([user, item], rating, batch_size=100, epochs=7, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
	model.save('model.h5')

	test_data = genfromtxt("test.csv", delimiter=',')
	test_user = test_data[1:,1].reshape((-1,1))
	test_item = test_data[1:,2].reshape((-1,1))
	ans = model.predict( [test_user,test_item], batch_size=100, verbose=0)

	with open(sys.argv[2],'w') as f:
		f.write('TestDataID,Rating\n')
		for idx,a in enumerate(ans):
			f.write('{},{}\n'.format(idx+1,(a[0]*r_std+r_mean)[0]))

if __name__=='__main__':
    main()

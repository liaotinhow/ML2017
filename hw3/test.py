import numpy as np
import sys
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.models import load_model


def load_test():
    test = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype='str')
    test = test[:,1]

    testdata = np.empty( (test.shape[0], 48, 48, 1))
    for i in range(test.shape[0]):
        tmp = np.asarray(test[i].split(' '), dtype=np.float32)
        testdata[i,:, :,:]= tmp.reshape((48, 48, 1))
    return testdata

emotion_classifier = load_model('model.h5')
emotion_classifier.summary()
test_gogo = load_test()
test_gogo = test_gogo/255
ans = emotion_classifier.predict_classes(test_gogo,batch_size=128)
with open(sys.argv[2],'w') as f:
        f.write('id,label\n')
        for idx,a in enumerate(ans):
            f.write('{},{}\n'.format(idx,a))
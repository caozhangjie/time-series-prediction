import numpy as np
import numpy.fft as myfft
import numpy.random as myrand
import numpy.matlib as nmatlib

train_data = np.load("../data/mean/n_train_data.npy")
test_data = np.load("../data/mean/n_test_data.npy")
train_label = train_data[:,0:1,0]
test_label = test_data[:,0:1,0]
train_data = train_data / 50.0
train_data[train_data > 2.0] = 2.0
train_data[train_data < -2.0] = -2.0
test_data = test_data / 50.0
test_data[test_data > 2.0] = 2.0
test_data[test_data < -2.0] = -2.0
train_feature = np.zeros(train_data.shape)
test_feature = np.zeros(test_data.shape)
for i in xrange(14):
    train_feature[:,:,i] = np.absolute(myfft.fft(train_data[:,:,i]))
    test_feature[:,:,i] = np.absolute(myfft.fft(test_data[:,:,i]))
train0 = train_feature.shape[0]
test0 = test_feature.shape[0]
train_feature = train_feature.reshape((train0, 14*20))
test_feature = test_feature.reshape((test0, 14*20))
train_feature_b = np.concatenate((train_feature, np.ones([train0, 1]).astype(np.float32)), axis=1)
test_feature_b = np.concatenate((test_feature, np.ones([test0, 1]).astype(np.float32)), axis=1)
weight_p = myrand.normal(0, 0.001, (14*20+1, 1))

iter_num = 20
lr = 0.000003
for i in xrange(iter_num):
    output = np.dot(train_feature_b, weight_p)
    probability = 1.0 / (1.0 + np.exp(-output))
    gradient = np.sum(np.multiply(nmatlib.repmat((train_label - probability), 1, 14*20+1), train_feature_b), axis=0).reshape(281, 1) / train0
    weight_p = weight_p - (gradient * lr)

predict_label = (1.0 / (1.0 + np.exp(-np.dot(test_feature_b, weight_p)))) < 0.5
accuracy = np.sum(np.abs(test_label - predict_label)) / float(test0)
print("accuracy: " + str(accuracy))

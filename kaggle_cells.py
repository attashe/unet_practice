import os
import sys
import random
import warnings

import cv2
import cPickle

import numpy as np
import pandas as pd
import lasagne
import theano
import theano.tensor as T

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain

from lasagne.layers import (InputLayer, ConcatLayer, Pool2DLayer, ReshapeLayer, DimshuffleLayer, NonlinearityLayer,
                            DropoutLayer, Deconv2DLayer, batch_norm)
from lasagne.layers import Conv2DLayer as ConvLayer
from collections import OrderedDict
from lasagne.init import HeNormal

def data_generator(x, y, batch_size=10):
    for i in range(len(x) / batch_size):
        yield (x[i*batch_size:i*batch_size+batch_size], 
               y[i*batch_size:i*batch_size+batch_size])     


TRAIN_DIR = 'E:/GitHub/kaggle_cells/data/stage1_train/'

train_ids = os.listdir(TRAIN_DIR)
train_images = [os.path.join(TRAIN_DIR, train_id, 'images', train_id + '.png') 
                for train_id in train_ids]
train_masks = {train_id: [os.path.join(TRAIN_DIR, train_id, 'masks', img_name) 
                          for img_name in os.listdir(os.path.join(TRAIN_DIR, train_id, 'masks'))]
               for train_id in train_ids}

X = {train_ids[i]: cv2.imread(train_images[i]) for i in range(len(train_images))}

Y = {train_id: sum((cv2.imread(train_mask)[..., 0]
                    for train_mask in train_masks[train_id]))
     for train_id in train_ids}

x = np.zeros((len(train_ids),) + X[train_ids[0]].shape)
y = np.zeros((len(train_ids),) + Y[train_ids[0]].shape)
# assert(X[train_ids[0]].shape == Y[train_ids[0]].shape)
print x.shape

for i, ind in zip(range(len(train_ids)), train_ids):
    x[i,:, :, :] =  cv2.resize(X[ind], (256, 256))
    y[i, :, :] = cv2.resize(Y[ind], (256, 256))


print [np.sum(y == 0), np.sum(y == 1)]
class_frequencies = np.array([np.sum(y == 0), np.sum(y == 1)])
# we will reweight the loss to put more focus on road pixels (because of class imbalance). This is a simple approach
# and could be improved if you also have a class imbalance in your experiments.
# we are taking **0.25 here because we want the net to focus more on the road pixels but not too much (otherwise
# it would not be penalized enough for missclassifying terrain pixels which results in too many false positives)
class_weights = (class_frequencies[[1,0]])**0.25
class_weights = class_weights / np.sum(class_weights) * 2.
class_weights = class_weights.astype(np.float32)



net = OrderedDict()
nonlinearity=lasagne.nonlinearities.elu
num_output_classes=2
n_input_channels=3
BATCH_SIZE=None
num_output_classes=2
pad='same'
nonlinearity=lasagne.nonlinearities.elu
input_dim=(256, 256)
base_n_filters=64
do_dropout=True

net['input'] = InputLayer((None, 3, 256, 256))

net['contr_1_1'] = batch_norm(ConvLayer(net['input'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
net['contr_1_2'] = batch_norm(ConvLayer(net['contr_1_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
net['pool1'] = Pool2DLayer(net['contr_1_2'], 2)

net['contr_2_1'] = batch_norm(ConvLayer(net['pool1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
net['contr_2_2'] = batch_norm(ConvLayer(net['contr_2_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
net['pool2'] = Pool2DLayer(net['contr_2_2'], 2)

net['contr_3_1'] = batch_norm(ConvLayer(net['pool2'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
net['contr_3_2'] = batch_norm(ConvLayer(net['contr_3_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
net['pool3'] = Pool2DLayer(net['contr_3_2'], 2)

net['contr_4_1'] = batch_norm(ConvLayer(net['pool3'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
net['contr_4_2'] = batch_norm(ConvLayer(net['contr_4_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
l = net['pool4'] = Pool2DLayer(net['contr_4_2'], 2)
# the paper does not really describe where and how dropout is added. Feel free to try more options
if do_dropout:
    l = DropoutLayer(l, p=0.4)

net['encode_1'] = batch_norm(ConvLayer(l, base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
net['encode_2'] = batch_norm(ConvLayer(net['encode_1'], base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
net['upscale1'] = batch_norm(Deconv2DLayer(net['encode_2'], base_n_filters*16, 2, 2, crop="valid", nonlinearity=nonlinearity, W=HeNormal(gain="relu")))

net['concat1'] = ConcatLayer([net['upscale1'], net['contr_4_2']], cropping=(None, None, "center", "center"))
net['expand_1_1'] = batch_norm(ConvLayer(net['concat1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
net['expand_1_2'] = batch_norm(ConvLayer(net['expand_1_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
net['upscale2'] = batch_norm(Deconv2DLayer(net['expand_1_2'], base_n_filters*8, 2, 2, crop="valid", nonlinearity=nonlinearity, W=HeNormal(gain="relu")))

net['concat2'] = ConcatLayer([net['upscale2'], net['contr_3_2']], cropping=(None, None, "center", "center"))
net['expand_2_1'] = batch_norm(ConvLayer(net['concat2'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
net['expand_2_2'] = batch_norm(ConvLayer(net['expand_2_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
net['upscale3'] = batch_norm(Deconv2DLayer(net['expand_2_2'], base_n_filters*4, 2, 2, crop="valid", nonlinearity=nonlinearity, W=HeNormal(gain="relu")))

net['concat3'] = ConcatLayer([net['upscale3'], net['contr_2_2']], cropping=(None, None, "center", "center"))
net['expand_3_1'] = batch_norm(ConvLayer(net['concat3'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
net['expand_3_2'] = batch_norm(ConvLayer(net['expand_3_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
net['upscale4'] = batch_norm(Deconv2DLayer(net['expand_3_2'], base_n_filters*2, 2, 2, crop="valid", nonlinearity=nonlinearity, W=HeNormal(gain="relu")))

net['concat4'] = ConcatLayer([net['upscale4'], net['contr_1_2']], cropping=(None, None, "center", "center"))
net['expand_4_1'] = batch_norm(ConvLayer(net['concat4'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
net['expand_4_2'] = batch_norm(ConvLayer(net['expand_4_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))

net['output_segmentation'] = ConvLayer(net['expand_4_2'], num_output_classes, 1, nonlinearity=None)
net['dimshuffle'] = DimshuffleLayer(net['output_segmentation'], (1, 0, 2, 3))
net['reshapeSeg'] = ReshapeLayer(net['dimshuffle'], (num_output_classes, -1))
net['dimshuffle2'] = DimshuffleLayer(net['reshapeSeg'], (1, 0))
net['output_flattened'] = NonlinearityLayer(net['dimshuffle2'], nonlinearity=lasagne.nonlinearities.softmax)

x_sym = T.tensor4()
seg_sym = T.ivector()
w_sym = T.vector()
    
output_layer_for_loss = net["output_flattened"]

# add some weight decay
l2_loss = lasagne.regularization.regularize_network_params(output_layer_for_loss, lasagne.regularization.l2) * 1e-4

# the distinction between prediction_train and test is important only if we enable dropout
prediction_train = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=False, batch_norm_update_averages=False, batch_norm_use_averages=False)
# we could use a binary loss but I stuck with categorical crossentropy so that less code has to be changed if your
# application has more than two classes
loss = lasagne.objectives.categorical_crossentropy(prediction_train, seg_sym)
loss *= w_sym
loss = loss.mean()
loss += l2_loss
acc_train = T.mean(T.eq(T.argmax(prediction_train, axis=1), seg_sym), dtype=theano.config.floatX)

prediction_test = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=True, batch_norm_update_averages=False, batch_norm_use_averages=False)
loss_val = lasagne.objectives.categorical_crossentropy(prediction_test, seg_sym)

# we multiply our loss by a weight map. In this example the weight map simply increases the loss for road pixels and
# decreases the loss for other pixels. We do this to ensure that the network puts more focus on getting the roads
# right
loss_val *= w_sym
loss_val = loss_val.mean()
loss_val += l2_loss
acc = T.mean(T.eq(T.argmax(prediction_test, axis=1), seg_sym), dtype=theano.config.floatX)

# learning rate has to be a shared variable because we decrease it with every epoch
params = lasagne.layers.get_all_params(output_layer_for_loss, trainable=True)
learning_rate = theano.shared(np.float32(0.001))
updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

# create a convenience function to get the segmentation
seg_output = lasagne.layers.get_output(net["output_segmentation"], x_sym, deterministic=True)
seg_output = seg_output.argmax(1)

train_fn = theano.function([x_sym, seg_sym, w_sym], [loss, acc_train], updates=updates)
val_fn = theano.function([x_sym, seg_sym, w_sym], [loss_val, acc])
get_segmentation = theano.function([x_sym], seg_output)
# we need this for calculating the AUC score
get_class_probas = theano.function([x_sym], prediction_test)

with open("E:\\UNet_params_ep.pkl", 'r') as f:
        params = cPickle.load(f)
        lasagne.layers.set_all_param_values(output_layer_for_loss, params)

N_EPOCHS = 20
N_BATCHES_PER_EPOCH = 320
for epoch in np.arange(0, N_EPOCHS):
    print epoch
    losses_train = []
    n_batches = 0
    accuracies_train = []
    print x.shape
    for data, target in data_generator(x.swapaxes(1, 3).swapaxes(1, 2), y.reshape((-1, 1, 256, 256))/255, 2):
#         print data.shape, target.shape
        # the output of the net has shape (BATCH_SIZE, N_CLASSES). We therefore need to flatten the segmentation so
        # that we can match it with the prediction via the crossentropy loss function
        target_flat = target.ravel()
        target_flat = target_flat.astype(np.int32, copy=False)
        loss, acc = train_fn(data.astype(np.float32), target_flat, class_weights[target_flat])
        losses_train.append(loss)
        accuracies_train.append(acc)
        n_batches += 1
        if n_batches > N_BATCHES_PER_EPOCH:
            break
    print "epoch: ", epoch, "\ntrain accuracy: ", np.mean(accuracies_train), " train loss: ", np.mean(losses_train)



with open("E:\\UNet_params_ep.pkl", 'w') as f:
            cPickle.dump(lasagne.layers.get_all_param_values(output_layer_for_loss), f)

import numpy as np
import time
import pickle
from snn_multipleneurons_fast import *
import sys
from sklearn.svm import LinearSVC
from mnist_data import get_mnist
import argparse

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--tanh_factors', type=str, default= "1", 
  help='tanh parameter g: tanh(gx), one per layer, seperated by comma', metavar='a,b,c')
parser.add_argument('--distance_parameter', type=str, 
  help='distance parameter to define the radius for the feedforward connection, one per layer, seperated by comma', metavar='a,b,c')
parser.add_argument('--stride', type=str, default= "2", 
  help='stride for the feedforward connection, one per layer, seperated by comma', metavar='a,b,c')
parser.add_argument('--gamma_factor', type=float, default=0.0, 
  help='feedback parameter')
parser.add_argument('--mult_factor', type=str, default= "1", 
   help='multiplication factor, one per layer, seperated by comma', metavar='a,b,c')
parser.add_argument('--NpSs', type=str, default= "4",
  help='neuron per site, one per layer, seperated by comma', metavar='a,b,c')
args = parser.parse_args()

tanh_factors = list(map(float, args.tanh_factors.split(',')))
distance_parameter= list(map(float, args.distance_parameter.split(',')))
stride= list(map(float, args.stride.split(',')))
gamma_factor= float(args.gamma_factor)
mult_factor  = list(map(float, args.mult_factor.split(',')))
NpSs = list(map(int, args.NpSs.split(',')))

lateral_distance = [0]*len(distance_parameter)

x_train, x_test, y_train, y_test = get_mnist()

image_dim = 28
channels = 1
strides = stride
distances = distance_parameter
distances_lateral = lateral_distance
tanh_factors = tanh_factors
layers = len(distance_parameter)
gamma = gamma_factor
mult_factors = mult_factor

network = deep_network_GPU(image_dim = image_dim, channels = channels, 
                           NpSs=NpSs, strides=strides, distances=distances, 
                           layers=layers, gamma=gamma, lr=5e-3, lr_floor = 1e-4, 
                           decay=0.5, distances_lateral = distances_lateral, 
                           tanh_factors = tanh_factors, mult_factors = mult_factors, euler_step=0.2)

print('NpS: ', network.NpSs)
print('Strides: ', network.strides)
print('Distances: ', network.distances)
print('Lateral Distances: ', network.lateral_distances)
print('gamma :', network.gamma)
print('tanh_factor: ', network.tanh_factors)

for i in range(120):
    indices = np.arange(0, x_train.shape[0])
    rand_indices = np.random.choice(indices, size=1000)
    x_train_rand = x_train[rand_indices]
    network.training(epochs=1, images=x_train_rand)

with open('network_g_val{0}_m{1}_g{2}_r{3}_NpS{4}_full.pkl'.format(tanh_factors, mult_factor, gamma_factor, distance_parameter, NpSs), 'wb') as output:
		    pickle.dump(network, output, pickle.HIGHEST_PROTOCOL)

#train_representations_output = np.zeros((x_train.shape[0], network.dimensions[-1])
train_representations = np.zeros((x_train.shape[0], np.sum(network.dimensions)))
for i in range(x_train.shape[0]):
    train_rep, _ = network.neural_dynamics(x_train[i])
    train_representations[i] = train_rep
    if (i+1)%1000==0:
        print(i+1)

np.save('train_representations_g_val{0}_m{1}_g{2}_r{3}_NpS{4}.npy'.format(tanh_factors, mult_factor, gamma_factor, distance_parameter, NpSs), train_representations)

test_representations = np.zeros((x_test.shape[0], np.sum(network.dimensions)))
for i in range(x_test.shape[0]):
    test_rep, _ = network.neural_dynamics(x_test[i])
    test_representations[i] = test_rep
    if (i+1)%1000==0:
        print(i+1)

np.save('test_representations_g_val{0}_m{1}_g{2}_r{3}_NpS{4}.npy'.format(tanh_factors, mult_factor, gamma_factor, distance_parameter, NpSs), test_representations)

clf_output = LinearSVC(random_state=0, tol=1e-5, max_iter = 1e6, class_weight='balanced')
clf_output.fit(train_representations[:,-network.dimensions[-1]:], y_train)
train_score = clf_output.score(train_representations[:, -network.dimensions[-1]:], y_train)
test_score = clf_output.score(test_representations[:, -network.dimensions[-1]:], y_test)

print('Test Score: ', test_score)
print('Train Score: ', train_score)

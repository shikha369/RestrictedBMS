#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import os
import cPickle
import gzip
import time as t


def load_data():
	# build the dataset and then split it into data
	# and labels
    datasetPath = ('train.csv')
    X = np.genfromtxt(datasetPath, delimiter = ",", dtype = "uint8")
    y = X[1:, 0]
    X = X[1:, 1:]
    X_test = X[0:10000,]
    y_test = y[0:10000,]
    X = X[10000:,]
    y = y[10000:,]
    datasetPath = ('test.csv')
    X_ = np.genfromtxt(datasetPath, delimiter = ",", dtype = "uint8")
    X_ = X_[1:,]
    '''this one has only train... u don't need labels so scrap labels here'''
    X_train = np.vstack((X,X_))
    low_values_indices =  X_train< 127
    X_train[low_values_indices] = 0
    high_values_indices = X_train>= 127
    X_train[high_values_indices] = 1
    low_values_indices =  X_test< 127
    X_test[low_values_indices] = 0
    high_values_indices = X_test>= 127
    X_test[high_values_indices] = 1
    return X_train, X_test, y_test


class RBM:

  def __init__(self, num_visible, num_hidden, learning_rate = 0.01):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.learning_rate = learning_rate
    self.wt_decay = 0.0002
    self.batch_size = 10

    self.weights = 2*np.random.randn(self.num_visible, self.num_hidden)/np.sqrt(self.num_visible +self.num_hidden)
    self.bias_vis = np.zeros((1, self.num_visible))
    self.bias_hid = -4*np.ones((1, self.num_hidden)) # -4 : crude method of encouraging sparsity


  def train(self, full_data,test_set_x, max_epochs = 2000):
    np.random.shuffle(full_data)

    held_out_data = full_data[0:1000,]
    train_data_representative = full_data[1000:2000,]
    full_data = full_data[1000:60000,]
    num_examples = full_data.shape[0]
    print num_examples
    batch_size = self.batch_size
    num_batches = num_examples / batch_size
    print num_batches
    if num_examples % batch_size != 0:
      num_batches += 1
    mom = 0.5 # Initial momentum
    algo = 'mom'
    delta_w = np.zeros((np.shape(self.weights)))
    delta_bias_hid = np.zeros((np.shape(self.bias_hid)))
    delta_bias_vis = np.zeros((np.shape(self.bias_vis)))

    moving_error = []
    arr_energy_held_out = []
    arr_train_rep = []
    for epoch in range(max_epochs):
      X = full_data
      np.random.shuffle(X)

      print "epoch "+str(epoch)
      epoch_error = []
      for j in xrange(0, num_batches):
          #print X.shape
          data = X[j * batch_size : (j + 1) * batch_size]

          # positive CD phase

          pos_hidden_activations = np.dot(data, self.weights)+ self.bias_hid
          pos_hidden_probs = self._logistic(pos_hidden_activations)

          pos_hidden_states = pos_hidden_probs > np.random.rand(batch_size, self.num_hidden)

          pos_associations = np.dot(data.T, pos_hidden_probs)
          bias_pos_hidd_act = np.sum(pos_hidden_probs,0)
          bias_pos_vis_act = np.sum(data,0)

          # "negative CD phase

          neg_visible_activations = np.dot(pos_hidden_states, self.weights.T) + self.bias_vis
          neg_visible_probs = self._logistic(neg_visible_activations)

          neg_hidden_activations = np.dot(neg_visible_probs, self.weights) + self.bias_hid
          neg_hidden_probs = self._logistic(neg_hidden_activations)

          neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
          bias_neg_hidd_act = np.sum(neg_hidden_probs,0)
          bias_neg_vis_act = np.sum(neg_visible_probs,0)

          # Update weights.
          if algo == 'mom':
              delta_w = mom*delta_w + self.learning_rate * ((pos_associations - neg_associations) / batch_size - self.wt_decay * self.weights)
              self.weights += delta_w
              delta_bias_hid = mom*delta_bias_hid + self.learning_rate *(bias_pos_hidd_act-bias_neg_hidd_act)/batch_size
              self.bias_hid+=delta_bias_hid
              delta_bias_vis = mom*delta_bias_vis + self.learning_rate*(bias_pos_vis_act-bias_neg_vis_act)/batch_size
              self.bias_vis += delta_bias_vis
          if algo == 'gd':
              self.weights += self.learning_rate * ((pos_associations - neg_associations) / batch_size - self.wt_decay * self.weights)
              self.bias_hid += self.learning_rate *(bias_pos_hidd_act-bias_neg_hidd_act)/batch_size
              self.bias_hid += self.learning_rate*(bias_pos_vis_act-bias_neg_vis_act)/batch_size

          error = np.sum((data - neg_visible_probs) ** 2)
          #print("Batch %s: error is %s" % (j, error))
          epoch_error.append(error)
      moving_error.append(np.mean(np.asarray(epoch_error)))
      t.sleep(1)
      print("Epoch %s: error is %s" % (epoch,np.mean(np.asarray(epoch_error)) ))
      """
	    may be not the best point to increase to increase momentum
	    if epoch > 500:
          mom = 0.9
          print "momentum increased to 0.9"
      """

      if (epoch % 20 == 0):
          print "saving hidden, error and weights !!!!!!!!!!!!!!!!!"
          moving_err = np.asarray(moving_error)
          np.savetxt("mov_error.txt",moving_err)
          np.savetxt("arr_energy_held_out.txt",arr_energy_held_out)
          np.savetxt("arr_train_rep.txt",arr_train_rep)
          h_rep=self.run_visible(test_set_x)
          np.savetxt("test_hid_rep"+str(epoch)+".txt",h_rep)
          np.savetxt("weights_learned"+str(epoch)+".txt",self.weights)

      if (epoch % 5 == 0):
          """check for energies"""
          energy_held_out = self.computeEnergy(held_out_data)
          energy_train_rep = self.computeEnergy(train_data_representative)
          arr_energy_held_out.append(energy_held_out)
          arr_train_rep.append(energy_train_rep)
    np.savetxt("arr_energy_held_out.txt",arr_energy_held_out)
    np.savetxt("arr_train_rep.txt",arr_train_rep)

  def computeEnergy(self, data):
      hidden_states = self.run_visible(data)
      energy = 0
      for i in xrange(0, 1000):
          energy += np.dot(data[i,:],np.dot(self.weights,hidden_states[i,:].T ))+ np.dot(self.bias_hid,hidden_states[i,:].T) + np.dot(self.bias_vis,data[i,:].T)
      return (-1 * energy)

  def run_visible(self, data):
    """
   Feeding test data to trained RBM to get its hidden representations
    """

    num_examples = data.shape[0]

    # sampled from a training example.
    hidden_states = np.ones((num_examples, self.num_hidden))
    # Calculate the activations of the hidden units.
    hidden_activations = np.dot(data, self.weights) + self.bias_hid
    # Calculate the probabilities of turning the hidden units on.
    hidden_probs = self._logistic(hidden_activations)
    # Turn the hidden units on with their specified probabilities.
    #hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden)
    #hidden_states = hidden_states[:,:]
    #return hidden_states
    return hidden_probs


  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))


if __name__ == '__main__':
  r = RBM(num_visible = 784, num_hidden = 500)
  train_set_x, test_set_x, test_set_y =load_data()
  r.train(train_set_x,test_set_x, max_epochs = 200)
  np.savetxt("main_weights_learned.txt",r.weights)
  """print r.weights"""
  h_rep=r.run_visible(test_set_x)
  np.savetxt("main_test_hid_rep.txt",h_rep)
  """ print h_rep"""


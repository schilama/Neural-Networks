import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N, F = X.shape #N = Number of data points, F = features
  C = W.shape[1]
  
  for i in range(N):
    scores = X[i].dot(W)
    maxval = np.max(scores)
    denominator = 0
    gradnumerator = []
    #Loop to compute loss
    for j in range(C): 
      scores[j] -= maxval
      if j == y[i]:
        numerator = np.exp(scores[j])
      denominator += np.exp(scores[j])
      gradnumerator.append(np.exp(scores[j]))
    #Loop to update gradients
    for j in range(C):
      if j == y[i]:
          dW[:,j] += X[i]*(-1.0 + (float(gradnumerator[j])/denominator))
          continue
      dW[:,j] += X[i]*(float(gradnumerator[j])/denominator)
    
    loss -= np.log(float(numerator)/denominator)
  
  loss *= (1.0/N)
  loss += 0.5 * reg * np.sum(W*W)
  
  dW *= (1.0/N)
  dW += reg * W
          
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0] #Number of data points
  C = W.shape[1] #Number of classes
  
  scores_v = X.dot(W) #N by C
  
  maxarr = np.max(scores_v,axis=1) #N
  maxarr = maxarr[:,np.newaxis] #N by 1 to support broadcasting
  scores_v = scores_v - maxarr #Stable computation
  
  rowindex = np.arange(N) #Select all data
  columnindex = y #Select only the true label class
  correct_scores_v = scores_v[rowindex,columnindex] #N, slice for score values of true label
  
  denominator = np.sum(np.exp(scores_v),axis=1) #N
  numerator = np.exp(correct_scores_v) #N
  loss = np.sum(-np.log(numerator) + np.log(denominator))*(1.0/N)
  loss += 0.5 * reg * np.sum(W*W)
  
  matrixforgrad = np.exp(scores_v)/denominator[:,np.newaxis]
  matrixforgrad[rowindex,columnindex] += -1.0 
  dW = np.dot(X.T,matrixforgrad)*(1.0/N)
  dW += reg * W
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        W1 = np.random.randn(input_dim, hidden_dim) * weight_scale
        b1 = np.zeros(hidden_dim) * weight_scale
        W2 = np.random.randn(hidden_dim,num_classes) * weight_scale
        b2 = np.zeros(num_classes) * weight_scale

        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        hidden_1 = np.maximum(X.dot(W1) + b1, 0)
        scores = hidden_1.dot(W2) + b2
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        

        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # get the parameters
        num_train = X.shape[0]
        input_dim = self.params['W1'].shape[0]
        hidden_dim = self.params['W1'].shape[1]
        num_classes = self.params['W2'].shape[1]
        
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']


        indicator = np.zeros((num_train, num_classes))
        for i in range(num_train):
            indicator[i, y[i]] = 1
        
        
        hidden_mat = np.maximum(X.dot(W1) + b1, 0)
        output = hidden_mat.dot(W2) + b2
        
        loss = - sum (np.log(np.sum(np.exp(output) * indicator, axis=1) / np.sum(np.exp(output), axis=1))) / num_train
        reg_value = np.sum(W1 * W1) + np.sum(W2 * W2)
        loss += 0.5 * self.reg * reg_value
         
        
        mat_2 = np.hstack([hidden_mat, np.ones((hidden_mat.shape[0],1))])
        W_2 = np.vstack([W2, b2 ])
        
        # for dW2, db2
        sum_ = np.sum(np.exp(scores), axis=1)
        df_softmax = ((np.exp(scores).transpose()/sum_).transpose() - indicator) / num_train
        dW_2 = mat_2.transpose().dot(df_softmax)
        dW_2[:-1,:] += self.reg * W_2[:-1,:]
        
        # for dW1, db1
        
        # dX2, where X2 means sencond layer input
        df_X2 = df_softmax.dot(W2.transpose())
        
        mat_1 = np.hstack([X, np.ones((X.shape[0],1))])
        indicator2 = X.dot(W1) + b1 > 0
        
        dW_1 = mat_1.transpose().dot(df_X2 * indicator2)
        W_1 = np.vstack([W1, b1])
        dW_1[:-1,:] += self.reg * W_1[:-1,:]

        grads['W1'] = dW_1[:-1,:]
        grads['b1'] = dW_1[-1,:]
        grads['W2'] = dW_2[:-1,:]
        grads['b2'] = dW_2[-1,:]


        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        lay_dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(len(lay_dims) - 1):
            dim_in = lay_dims[i]
            dim_out = lay_dims[i+1]
            
            params_name1 = "W{}".format(i+1)
            params_name2 = "b{}".format(i+1)

            self.params[params_name1] = np.random.randn(dim_in, dim_out) * weight_scale
            self.params[params_name2] = np.zeros(dim_out)
            
            if self.normalization == "batchnorm":
                params_name3 = "gamma{}".format(i)
                params_name4 = "beta{}".format(i)

                self.params[params_name3] = np.ones(dim_out)
                self.params[params_name4] = np.zeros(dim_out)
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.reshape(X.shape[0], -1)
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        terms = X
        reg_value = 0
        i = 1
        forward_record = [terms.copy()]
        dropout_record = []
        if self.dropout_param.get("seed") is not None:
            np.random.seed(self.dropout_param["seed"])

        while 1:

            W_name = "W{}".format(i)
            b_name = "b{}".format(i)
            W = self.params.get(W_name)
            b = self.params.get(b_name)   
            if W is None:               
                break
            else:
                if i > 1: 
                    terms = np.maximum(terms, 0)
                    if self.use_dropout:
                        mask = np.random.rand(*terms.shape) < self.dropout_param['p']
                        terms *= mask / self.dropout_param["p"]
                        dropout_record.append(mask.copy())
                    forward_record.append(terms.copy())
                terms = terms.dot(W) + b
                
                #
                reg_value += np.sum(W * W)
                i += 1
        
        scores = terms
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        num_train = scores.shape[0]

        indicator = np.zeros_like(scores)
        for i in range(num_train):
            indicator[i, y[i]] = 1 

        loss = - sum (np.log(np.sum(np.exp(scores) * indicator, axis=1) / np.sum(np.exp(scores), axis=1))) / num_train
        loss += 0.5 * self.reg * reg_value

        sum_ = np.sum(np.exp(scores), axis=1)
        df_softmax = ((np.exp(scores).transpose()/sum_).transpose() - indicator) / num_train
        d_upstream = df_softmax
        i = len(forward_record) - 1
        while i >= 0:
            XX = forward_record[i]
            WW = self.params["W{}".format(i+1)]
            bb = self.params["b{}".format(i+1)]
            if i < len(forward_record) - 1:
                indicator = XX.dot(WW) + bb > 0
            else:
                indicator = np.ones((XX.shape[0], WW.shape[1]))
            
            d_W = XX.transpose().dot(d_upstream * indicator)
            d_W += self.reg * WW
            d_b = np.sum(d_upstream * indicator, axis=0)
            grads["W{}".format(i+1)] = d_W
            grads["b{}".format(i+1)] = d_b
            d_upstream = (d_upstream * indicator).dot(WW.T)
            if self.use_dropout and len(dropout_record) > 0:
                mask = dropout_record.pop()
                d_upstream *= mask / self.dropout_param["p"]
            i -= 1

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

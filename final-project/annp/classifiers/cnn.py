from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1']=np.random.normal(scale=weight_scale,size=[num_filters,input_dim[0],filter_size,filter_size])
        self.params['b1']=np.zeros(num_filters)
        self.params['W2']=np.random.normal(scale=weight_scale,size=[num_filters*input_dim[1]*input_dim[2]//4,hidden_dim])
        self.params['b2']=np.zeros(hidden_dim)
        self.params['W3']=np.random.normal(scale=weight_scale,size=[hidden_dim,num_classes])
        self.params['b3']=np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in annp/fast_layers.py and  #
        # annp/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        scores_1,cache_1=conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
        scores_2=np.reshape(scores_1,(X.shape[0],-1))
        scores_3,cache_3=affine_relu_forward(scores_2,W2,b2)
        scores,cache=affine_forward(scores_3,W3,b3)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss,dscores=softmax_loss(scores,y)
        loss += 0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3))
        dscores_3,dW3,db3=affine_backward(dscores,cache)
        dscores_2,dW2,db2=affine_relu_backward(dscores_3,cache_3)
        dscores_1=np.reshape(dscores_2,scores_1.shape)
        _,dW1,db1=conv_relu_pool_backward(dscores_1,cache_1)
        grads['W1']=dW1+self.reg*W1
        grads['b1']=db1
        grads['W2']=dW2+self.reg*W2
        grads['b2']=db2
        grads['W3']=dW3+self.reg*W3
        grads['b3']=db3

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class ConvNet(object):
    def __init__(self, input_dim=(3,32,32), num_filters=32, filter_size=3,
                conv_layers=2, affine_layers=2, hidden_dims=[100, 100], num_classes=10, 
                weight_scale=1e-3, reg=0.0, dtype=np.float32, use_batchnorm=False):
        # 初始化参数并定义维度
        self.params = {}
        self.bn_params = {}
        self.spatialbn_params = {}
        self.use_bn = use_batchnorm
        self.reg = reg
        self.dtype = dtype
        self.filter_size = filter_size
        self.conv_layers = conv_layers
        self.affine_layers = affine_layers
        C,H,W = input_dim
        F = filter_size
        pad = (F-1)//2
        pool_S = 2
        H2,W2 = H,W

        # 初始化卷积层的权重、偏置量和归一化的参数(gamma,beta,均值与方差)
        for l in range(2*conv_layers):
            self.params['W%d' % (l+1)] = np.random.randn(num_filters,C,F,F) * weight_scale
            self.params['b%d' % (l+1)] = np.zeros(num_filters)
            C = num_filters
            if(l % 2 == 1 and l != 0):
                H2,W2 = [(H2+2*pad-F)//pool_S + 1, (W2+2*pad-F)//pool_S + 1]
            if(self.use_bn is True):
                self.params['gamma%d' % (l+1)] = np.ones(C)
                self.params['beta%d' % (l+1)] = np.zeros(C)
                self.spatialbn_params['running_mean'] = np.zeros(num_filters)
                self.spatialbn_params['running_var'] = np.zeros(num_filters)
      
        # 和上面类似，接下来初始化纺射层的参数（权重、偏置量、gamma 、beta以及均值与方差）
        if(affine_layers == 1):
            hidden_dims = [H2*W2*num_filters] + [hidden_dims] + [num_classes]
        else:
            hidden_dims = [H2*W2*num_filters] + hidden_dims + [num_classes]
            
        for l in range(affine_layers+1):
            self.params['W%d' % (l+2*conv_layers+1)] = np.random.randn(hidden_dims[l],hidden_dims[l+1]) * weight_scale
            self.params['b%d' % (l+2*conv_layers+1)] = np.zeros(hidden_dims[l+1])
            if(self.use_bn is True and l != affine_layers):
                self.params['gamma%d' % (l+2*conv_layers+1)] = np.ones(hidden_dims[l+1])
                self.params['beta%d' % (l+2*conv_layers+1)] = np.zeros(hidden_dims[l+1])
                self.bn_params['running_mean'] = np.zeros(hidden_dims[l+1])
                self.bn_params['running_var'] = np.zeros(hidden_dims[l+1])

        # 将参数类型转变为浮点型
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        
    
    def loss(self, X, y=None):
        # 对卷积层和池化层的前向传播的一些参数赋值
        filter_size = self.filter_size
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        conv_cache = {}
        fc_cache = {}
        scores = None
        N,num_F,H,W = [0,0,0,0]
        conv_a = X
        conv_layers = self.conv_layers
        affine_layers = self.affine_layers
        fp_check = False
        use_bn = self.use_bn
        spatialbn_params = self.spatialbn_params
        bn_params = self.bn_params
        use_bn = self.use_bn
        if(use_bn == True):
            if(y is None):
                bn_params['mode'] = 'test'
                spatialbn_params['mode'] = 'test'
            else:
                bn_params['mode'] = 'train'
                spatialbn_params['mode'] = 'train'
        
        # 计算出前向传播的层数
        total_layers = conv_layers+affine_layers+1
        
        # 下面进行前向传播
        for l in range(total_layers):
            # 卷积层
            if(l < conv_layers): 
                conv_a, conv_cache[2*l+1] = conv_relu_forward(
                    conv_a, self.params['W%d' % (2*l+1)], self.params['b%d' % (2*l+1)], conv_param)
                conv_a, conv_cache[2*l+2] = conv_relu_pool_forward(
                    conv_a, self.params['W%d' % (2*l+2)], self.params['b%d' % (2*l+2)], conv_param, pool_param)
                if(l == conv_layers-1):
                    fp_check = True
            # 纺射层
            else:
                if(fp_check is True):
                    fp_check = False
                    N,num_F,H,W = conv_a.shape
                    fc_a = np.reshape(conv_a,(N,num_F*H*W))
                if(l == total_layers-1):
                    scores, fc_cache[l] = affine_forward(
                        fc_a, self.params['W%d' % (2*conv_layers+l-1)], self.params['b%d' % (2*conv_layers+l-1)])
                else:
                    fc_a, fc_cache[l] = affine_forward(
                        fc_a, self.params['W%d' % (2*conv_layers+l-1)], self.params['b%d' % (2*conv_layers+l-1)])
        if y is None:
            return scores
        
        loss, grads = 0, {}
        
        # 使用softmax计算loss
        loss, delta_l = softmax_loss(scores,y)
        
        # 反向传播
        for l in range(total_layers-1,-1,-1):
            # 纺射层
            if(l >= conv_layers):                
                delta_l, grads['W%d' % (2*conv_layers+l-1)], grads['b%d' % (2*conv_layers+l-1)] = affine_backward(delta_l, fc_cache[l])
                if(l == conv_layers):
                    delta_l = np.reshape(delta_l, (N,num_F,H,W))
            # 卷积层
            else:
                delta_l, grads['W%d' % (2*l+2)], grads['b%d' % (2*l+2)] = conv_relu_pool_backward(delta_l, conv_cache[2*l+2])
                delta_l, grads['W%d' % (2*l+1)], grads['b%d' % (2*l+1)] = conv_relu_backward(delta_l, conv_cache[2*l+1])
        
        for l in range(total_layers):
            if(l < conv_layers):
                W1 = self.params['W%d' % (2*l+1)]
                W2 = self.params['W%d' % (2*l+2)]
                loss += 0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2))
                grads['W%d' % (2*l+1)] += self.reg*W1
                grads['W%d' % (2*l+2)] += self.reg*W2
            else:
                W = self.params['W%d' % (2*conv_layers+l-1)]
                loss += 0.5*self.reg*np.sum(W*W)
                grads['W%d' % (2*conv_layers+l-1)] += self.reg*W

        return loss, grads
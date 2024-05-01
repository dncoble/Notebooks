#%% load mnist
import numpy as np
import sklearn as sk
from sklearn import datasets
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl

mnist = sk.datasets.fetch_openml('mnist_784', as_frame=False, parser='liac-arff')

data=mnist['data']
target = np.asarray(mnist['target'],dtype=int)

# use 1/10 of data
data = data[::10,:]
target = target[::10]


target_ = np.zeros((data.shape[0], 10), dtype=bool)
for i, d in enumerate(target):
    target_[i,d] = True
target = target_

data_train, data_test, target_train, target_test = sk.model_selection.\
    train_test_split(data, target, test_size=0.2)
#%%
# plot a digit

digit_id = np.random.randint(0, high=data.shape[0])
test_digit = data[digit_id,:]
digit_reshaped = np.reshape(test_digit,(28,28))

plt.figure()
plt.imshow(digit_reshaped, cmap=mpl.cm.binary, interpolation='nearest')
plt.show()
#%%
nodes_layer1 = 10240
nodes_layer2 = 2560
nodes_layer3 = 640
nodes_layer4 = 160
nodes_layer5 = 40
nodes_layer6 = 10

nodes_per_layer = [nodes_layer1, nodes_layer2, nodes_layer3, nodes_layer4, nodes_layer5, nodes_layer6]
n_layers = len(nodes_per_layer)

# weights of each layer
# w_layer = [np.random.rand(nodes) > .5 for nodes in nodes_per_layer]
w_layer = [np.ones(nodes, dtype=bool) for nodes in nodes_per_layer]
w_reverse = [i for i in reversed(w_layer)]
# connections of each node
prev_nodes = [784] + nodes_per_layer[:-1]
c_layer = [np.random.randint(0, high=prev, size=(3, nodes)) for prev, nodes in zip(prev_nodes, nodes_per_layer)]
c_reverse = [i for i in reversed(c_layer)]
# negations of each node
n_layer = [np.random.rand(nodes, 3) > .5 for nodes in nodes_per_layer]
n_reverse = [i for i in reversed(n_layer)]

layer_1_threshold = np.random.rand(nodes_layer1, 3)

stop_training = False
history = {
    'error': [],
    'n_nodes_improve': [],
    'expected_next_error': [],
}
mask_array = np.array([[False, True, True], [True, False, True], [True, True, False]])
epoch = 0
N = data_train.shape[0]
# begin training
while(not stop_training):
    # forward inference
    x = data_train.T
    z_forward = []
    y_forward = []
    for i, (w, c, n) in enumerate(zip(w_layer, c_layer, n_layer)):
        if(i == 0):
            # https://numpy.org/doc/stable/user/basics.indexing.html
            z = x[c].T > layer_1_threshold
            z = np.logical_xor(z, n)
            y = np.all(z, axis=2)
            x = w*y
            x = x.T
            z_forward.append(z.copy())
            y_forward.append(y.copy())
        else:
            z = x[c].T
            z = np.logical_xor(z, n)
            y = np.all(z, axis=2)
            x = w*y
            x = x.T
            z_forward.append(z.copy())
            y_forward.append(y.copy())
    
    # calculate error
    e = x != (target_train.T)
    E = 1/N * np.sum(e)
    print(E)
    
    # dx: increase in error if value is flipped
    dx = (1/(10*N))*e + (-1/(10*N))*(np.logical_not(e))
    # backpropagation
    z_reverse = [i for i in reversed(z_forward)]
    y_reverse = [i for i in reversed(y_forward)]
    dw_list = []
    for (i, w, c, n, z, y) in zip(range(5, -1, -1), w_reverse, c_reverse, n_reverse, z_reverse, y_reverse):
        du = dx
        dw = np.sum(du*y.T, axis=1)
        dw_list = [dw] + dw_list
        dx = np.zeros((nodes_per_layer[i-1], du.shape[-1]))
        if(i != 0): # no nead to get grad for inputs
            for j in range(dw.shape[0]):
                if(w[j]):
                    z_ = np.expand_dims(z[:,j,:], -1)
                    z_ = np.tile(z_, (1,1,3))[:,mask_array].reshape(-1, 3, 2)
                    z_ = np.all(z_, axis=-1)
                    for k in range(3):
                        dx[c[k, j]] += z_[:,k]*du[j]
    
    
    # update only one weight
    # dw_min = 0
    # ll = 0
    # mm = 0
    # for l in range(n_layers):
    #     prep_min = np.min(dw_list[l])
    #     if(prep_min < dw_min):
    #         dw_min = prep_min
    #         ll = l
    #         mm = np.argmin(dw_list[l])
    # if(dw_min == 0):
    #     stop_training = True
    # else:
    #     print('dw_min:', dw_min)
    #     print('layer of weight:', ll)
    #     print('index of weight:', mm)
    #     w_layer[ll][mm] = not w_layer[ll][mm]
    
    # update all weights with negative gradient
    any_negative = False
    n_weights_flipped = 0
    for l in range(n_layers):
        for m in range(nodes_per_layer[l]):
            if(dw_list[l][m] < 0):
                dw_list[l][m] = not dw_list[l][m]
                any_negative = True
                n_weights_flipped += 1
                print('l: ', l, ', m:', m)
    if(not any_negative):
        stop_training = True
    else:
        print(n_weights_flipped)
        
    
    
        
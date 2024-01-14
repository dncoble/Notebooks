#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import perf_counter
# create dataset

dp = 200000 # number of points in dataset
T_range = (5, 30) # range of periods
L_range = (10, 100) # range of how long each period will last.

X = np.zeros((dp))
y = np.zeros((dp))

T_current = T_range[0] + np.random.rand()*(T_range[1] - T_range[0])
j = np.random.randint(L_range[0], high=L_range[1])
theta = 0
for i in range(dp):
    X[i] = np.sin(theta)
    y[i] = T_current
    theta += 2*np.pi/T_current
    j -= 1
    if(j == 0):
        T_current = T_range[0] + np.random.rand()*(T_range[1] - T_range[0])
        j = np.random.randint(L_range[0], high=L_range[1])

# y = 1/y # predict frequency
X = X.reshape(-1, 1)
#%% Plot a random section of the dataset
j = np.random.randint(0, high=dp-100)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(y[j:j+100])
ax1.set_ylim((T_range[0], T_range[1]))
ax2.plot(X[j:j+100])
ax2.set_xlim((0, 100))
#%% RNN implementation
# RNN plus dense layer on top
class RNN:
    
    # initialize RNN
    def __init__(self, units, input_units, output_units):
        self.units = units
        self.input_shape = input_units
        
        self.W = (np.random.rand(units, input_units) - .5)*.05
        self.U = (np.random.rand(units, units) - .5)*.05
        self.b = (np.random.rand(units, 1) - .5)*.05
        self.d = (np.random.rand(output_units, units) - .5)*.05
        
        # state vector
        self.h = np.zeros((units, 1))
    
    def reset_h(self):
        self.h = np.zeros((self.units, 1))
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-1*x))
    
    # not used 
    def ddsigmoid(self, x):
        y = np.exp(-1*x)
        return y/(1+y)**2
    
    # calculate forward step and return all intermediate values
    def step_forward(self, x):
        z = self.W@x + self.U@self.h + self.b
        h = self.sigmoid(z)
        y_hat = self.d@h
        
        self.h = h.copy()
        return [y_hat, h, z, x]
    
    '''
    calculate gradient backwards and return all grads wrt weights
    intermediate_vars: y_hat, h, z, x at the calculated timestep
    '''
    def grad_backwards(self, y, de_dh1, *intermediate_vars):
        [y_hat, h, z, x] = intermediate_vars
        e = (y-y_hat)**2
        de_dy = -2*(y_ - y_hat)
        
        de_dh0 = (self.d*de_dy).T
        de_dh = de_dh0 + de_dh1
        de_dz = de_dh*(h - h**2) # this trick means we don't have to use z
        de_dW = de_dz@(x_.T)
        de_dU = de_dz@(h.T)
        de_db = de_dz
        de_dd = de_dy * h.T
        # backwards connection vector
        de_dh1 = (self.U.T)@(de_dz)
        
        return (de_dW, de_dU, de_db, de_dd, de_dh)
#%% standard first forward then backward
units = 300
epochs = 100
N = 50 # backpropagation length
lr = .0001 # learning rate

rnn = RNN(units, 1, 1)
#%% training
h_mat = np.zeros((N, units, 1))
de_dh_mat = np.zeros((N, units, 1))
y_hat_mat = np.zeros((N))
s1 = 0
s2 = 0
s3 = 0
s4 = 0

for epoch in range(1, epochs+1):
    print('beginning epoch %d'%epoch)
    y_pred = np.zeros(dp)
    # batches
    rnn.reset_h()
    for i, batch in tqdm(zip(range(0, dp, N), range(1, dp//N+1))):
        # forward pass
        t1 = perf_counter()
        j_iter = range(0, N)
        for i_, j in zip(range(i,i+N), j_iter):
            t1 = perf_counter()
            x_ = X[i_:i_+1,:]
            y_ = y[i_]
            
            [y_hat, h, z, x] = rnn.step_forward(x_)
            
            h_mat[j] = h
            y_hat_mat[j] = y_hat[0,0]
            y_pred[i] = y_hat[0,0]
        s1 += perf_counter() - t1
        # backprop
        t2 = perf_counter()
        delta_W = np.zeros((units, 1))
        delta_U = np.zeros((units, units))
        delta_b = np.zeros((units, 1))
        delta_d = np.zeros((1, units))
        de_dh1 = np.zeros((units, 1))
        z_ = np.zeros((units, 1)) # placeholder, not used in calculations
        j_iter = range(N-1, -1, -1)
        for i_, j in zip(range(i+N-1,i-1,-1), j_iter):
            t3 = perf_counter()
            y_ = y[i_]
            x_ = X[i_:i_+1,:].copy()
            y_hat = y_hat_mat[j].copy()
            h = h_mat[j]
            z = z_
            intermediate_vars = [y_hat, h, z, x]
            s3 += perf_counter() - t3
            
            t4 = perf_counter()
            (de_dW, de_dU, de_db, de_dd, de_dh1) = rnn.grad_backwards(y_, de_dh1, *intermediate_vars)
            de_dh_mat[j] = de_dh1
            s4 += perf_counter() - t4
            
            delta_W += 1/N*de_dW
            delta_U += 1/N*de_dU
            delta_b += 1/N*de_db
            delta_d += 1/N*de_dd
        s2 += perf_counter() - t2
        rnn.W -= lr*delta_W
        rnn.U -= lr*delta_U
        rnn.b -= lr*delta_b
        rnn.d -= lr*delta_d
        # find error across epoch
    ep = np.sqrt(np.mean((y_pred - y)**2))
    print('error: %f'%ep)
#%% predict entire dataset
y_pred = np.zeros(dp)
rnn.reset_h()
h_mat = np.zeros((dp, units, 1))
for i in range(dp):
    x_ = X[i:i+1,:]
    [y_hat, h, _, _] = rnn.step_forward(x_)
    h_mat[i] = h
    y_pred[i] = y_hat[0,0]
#%% plot some random section
j = np.random.randint(0, high=dp-100)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.set_ylim([0, 35])
ax1.plot(y[j:j+100])
ax1.plot(y_pred[j:j+100])
ax2.plot(X[j:j+100])
ax2.set_xlim((0, 100))
#%% using keras to see if RNNs are just bad
import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([
    keras.layers.LSTM(units,
                        activation='sigmoid',
                        stateful=True,
                        batch_input_shape=[64, None, 1],
                        return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
])
optimizer = keras.optimizers.SGD(
    learning_rate = lr,
    momentum = 0
)
model.compile(
    optimizer,
    loss='mse'
)
X_train = X.reshape(-1, N, 1)
y_train = y.reshape(-1, N, 1)

# j = np.random.randint(0, high=dp-100)
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.plot(y_train[1])
# ax1.set_ylim((T_range[0], T_range[1]))
# ax2.plot(X_train[1])

model.fit(X_train, y_train,
          epochs=20,
          batch_size=1,
          shuffle=False
)
#%%
y_pred = model.predict(X.reshape(1, -1, 1))

plt.figure()
plt.plot(y)
plt.plot(y_pred.flatten())
#%% forward-backward implementation
# h_mat = np.zeros((N, units))
# y_hat_mat = np.zeros((N, 1))
# y_mat = np.zeros((N, 1)) # easier for keeping track of indices but not required
# x_mat = np.zeros((N, 1)) # easier for keeping track of indices but not required
# h = np.zeros((units, 1))
# y_hat_all_time = np.zeros((dp))

# for epoch in range(1, epochs+1):
#     write_mode = True # true for forward, false for backward
#     for i in range(0, dp, N): # i: current timestep
#         if(write_mode):
#             h_iter = range(0, N)
#         else:
#             h_iter = range(N-1, -1, -1)
#         i_iter = range(i, i+N)
        
#         de_dh_future = np.zeros((units, 1))
        
#         delta_W = np.zeros((units, 1))
#         delta_U = np.zeros((units, units))
#         delta_b = np.zeros((units, 1))
#         delta_d = np.zeros((1, units))
#         for i, h_index in zip(i_iter, h_iter):
#             # forward pass
#             x = X[i:i+1,:]
#             y = y[i]
            
            
#             # backward pass
            
#             # pop matrix elements before replacing them
#             y_hat_bptt = y_hat_mat[h_index]
#             y_bptt = y_mat[h_index]
#             h_bptt = h_mat[h_index:h_index+1,:].T
#             x_bptt = x_mat[h_index]
            
#             y_hat_mat[h_index] = y_hat
#             y_mat[h_index] = y_.flatten()
#             h_mat[h_index] = h.flatten()
#             x_mat[h_index] = x_.flatten()
            
#             # backward section
#             y_hat = y_hat_bptt
#             y_ = y_bptt
#             h = h_bptt
#             x_ = x_bptt.reshape((1,1))
            
#             delta_W += de_dW
#             delta_U += de_dU
#             delta_b += de_dz # de_db = de_dz
#             delta_d += h.T # de_dd = h^T
            
#         # weight updating
#         W -= lr*delta_W
#         U -= lr*delta_U
#         b -= lr*delta_b
#         d -= lr*delta_d
#         # reverse write direction
#         write_mode = not write_mode
#     print('finished epoch %d'%epoch)
# #%% plot some training results
# j = np.random.randint(0, high=dp-100)
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.plot(y[j:j+100])
# ax1.plot(y_hat_all_time[j:j+100])
# ax2.plot(X[j:j+100])
# ax2.set_xlim((0, 100))
# #%%
# # test
# class Jip:
    
#     def __init__(self):
#         self.x = np.array([0])
        
#     def get_x(self):
#         return self.x
    
#     def change_x(self):
#         self.x += 1


# jip = Jip()

# x = jip.get_x()
# jip.change_x()
# #%%
# def hello(*args):
    
#     [a, b] = args
    
#     print(a)
#     print(b)

# hello('hello', 'world')
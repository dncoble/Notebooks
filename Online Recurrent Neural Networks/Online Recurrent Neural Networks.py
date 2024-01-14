#%%
import numpy as np
import matplotlib.pyplot as plt
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

X = X.reshape(-1, 1)
#%% Plot a random section of the dataset
j = np.random.randint(0, high=dp-100)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(y[j:j+100])
ax1.set_ylim((T_range[0], T_range[1]))
ax2.plot(X[j:j+100])
ax2.set_xlim((0, 100))
#%% Simultaneous forward pass and training
units = 50
epochs = 3
N = 100 # backpropagation length
lr = .01 # learning rate
# batches = dp//N
def sigmoid(x):
    return 1/(1+np.exp(-1*x))
def ddsigmoid(x):
    y = np.exp(-1*x)
    return y/(1+y)**2

# instantiate model
W = (np.random.rand(units, 1) - .5)*.05
U = (np.random.rand(units, units) - .5)*.05
b = (np.random.rand(units, 1) - .5)*.05
d = (np.random.rand(1, units) - .5)*.05

h_mat = np.zeros((N, units))
y_hat_mat = np.zeros((N, 1))
y_mat = np.zeros((N, 1)) # easier for keeping track of indices but not required
x_mat = np.zeros((N, 1)) # easier for keeping track of indices but not required
h = np.zeros((units, 1))
y_hat_all_time = np.zeros((dp))


for epoch in range(1, epochs+1):
    write_mode = True # true for forward, false for backward
    for i in range(0, dp, N): # i: current timestep
        if(write_mode):
            h_iter = range(0, N)
        else:
            h_iter = range(N-1, -1, -1)
        i_iter = range(i, i+N)
        
        de_dh_future = np.zeros((units, 1))
        
        delta_W = np.zeros((units, 1))
        delta_U = np.zeros((units, units))
        delta_b = np.zeros((units, 1))
        delta_d = np.zeros((1, units))
        for i, h_index in zip(i_iter, h_iter):
            # forward section
            x_ = X[i:i+1,:]
            y_ = y[i]
            
            h = sigmoid(W@x_ + U@h + b)
            y_hat = d@h
            y_hat_all_time[i] = y_hat[0,0]
            
            # pop matrix elements before replacing them
            y_hat_bptt = y_hat_mat[h_index]
            y_bptt = y_mat[h_index]
            h_bptt = h_mat[h_index:h_index+1,:].T
            x_bptt = x_mat[h_index]
            
            y_hat_mat[h_index] = y_hat
            y_mat[h_index] = y_.flatten()
            h_mat[h_index] = h.flatten()
            x_mat[h_index] = x_.flatten()
            
            # backward section
            y_hat = y_hat_bptt
            y_ = y_bptt
            h = h_bptt
            x_ = x_bptt.reshape((1,1))
            
            de_dy = -2/N*(y_ - y_hat)
            de_dh_current = (d*de_dy).T
            de_dh = de_dh_current + de_dh_future
            de_dz = de_dh*(h - h**2)
            de_dW = de_dz@(x_.T)
            de_dU = de_dz@(h.T)
            de_dh_future = (U.T)@(de_dz)
            
            delta_W += de_dW
            delta_U += de_dU
            delta_b += de_dz # de_db = de_dz
            delta_d += h.T # de_dd = h^T
            
        # weight updating
        W -= lr*delta_W
        U -= lr*delta_U
        b -= lr*delta_b
        d -= lr*delta_d
        # reverse write direction
        write_mode = not write_mode
    print('finished epoch %d'%epoch)
#%% plot some training results
j = np.random.randint(0, high=dp-100)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(y[j:j+100])
ax1.plot(y_hat_all_time[j:j+100])
ax2.plot(X[j:j+100])
ax2.set_xlim((0, 100))
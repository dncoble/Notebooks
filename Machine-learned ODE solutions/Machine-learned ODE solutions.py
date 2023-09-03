import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# tf.compat.v1.disable_eager_execution()
"""
tensorflow/keras classes for physics-informed machine learning
"""
#%% compute at interior, initial, and boundary
# needed for dispatching
class GradientLayer(keras.layers.Layer):
    
    def __init__(self,
               trainable=True,
               name=None,
               dtype=None,
               dynamic=False,
               **kwargs):
        super().__init__(trainable=True,
                       name=None,
                       dtype=None,
                       dynamic=False,
                       **kwargs)
    
    def call(self, inputs):
        y = inputs[0]
        x = inputs[1]
        return tf.gradients(y, x)[0]

class ModelInterior(keras.layers.Layer):
    
    '''
    wraps model and computes for interior
    '''
    def __init__(self, model,
               trainable=True,
               name=None,
               dtype=None,
               dynamic=False,
               **kwargs):
        self.model = model
        super().__init__(trainable=True,
                       name=None,
                       dtype=None,
                       dynamic=False,
                       **kwargs)
    
    def call(self, inputs):
        x = inputs[0]
        t = inputs[1]
        xt = tf.concat([x, t], -1)
        return self.model(xt)
        
class ModelInitial(keras.layers.Layer):
    '''
    wraps model and computes for only t=0
    '''
    def __init__(self, model, 
                 trainable=True,
                 name=None,
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        self.model = model
        super().__init__(trainable=True,
                       name=None,
                       dtype=None,
                       dynamic=False,
                       **kwargs)
    
    '''
    returns also t (needed for gradient calculation downstream)
    '''
    def call(self, inputs):
        x = inputs
        t = tf.zeros(x.shape)
        xt = tf.concat([x, t], -1)
        
        return [self.model(xt), t]

class ModelBoundary(keras.layers.Layer):
    '''
    wraps model and computes for x=0, x=1
    '''
    def __init__(self, model,
                 trainable=True,
                 name=None,
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        self.model = model
        super().__init__(trainable=True,
                       name=None,
                       dtype=None,
                       dynamic=False,
                       **kwargs)
    
    '''
    returns at x=0 and x=1
    '''
    def call(self, inputs):
        t = inputs
        x0 = tf.zeros(t.shape)
        x1 = tf.ones(t.shape)
        x0t = tf.concat([x0, t], -1)
        x1t = tf.concat([x1, t], -1)
        
        w0 = self.model(x0t)
        w1 = self.model(x1t)
        
        return [w0, w1]

#%% losses
# @tf.function
def wave_loss(w_interior, x_interior, t_interior, lam):
    grad1 = tf.gradients(w_interior, [x_interior, t_interior])
    grad2 = tf.gradients(grad1, [x_interior, t_interior])
    return tf.square(lam*grad2[0] - grad2[1])

# g1 - initial position
@tf.function
def g1(x):
    M = np.pi
    return tf.math.sin(M*x)
# g2 - initial velocity
@tf.function
def g2(x):
    return 0

def initial_loss(w_initial, x_initial, t_initial):
    g1x = g1(x_initial)
    # g2x = g2(x_initial)
    dwdt = tf.gradients(w_initial, t_initial)
    return tf.square(w_initial - g1x) + tf.square(dwdt)

def boundary_loss(w_bound1, w_bound2):
    return tf.square(w_bound1) + tf.square(w_bound2)

# physical constants
T = 1
L = 1
lam = (2*pi)**2

rhof = 1
rho0 = 1
rhob = 1
batch_size = 128
model_shape = [30, 30, 1]
# number of points at initial, boundary, and interior during each epoch. Each must be a multiple of the batch size. 
Nf = 8192 # number of interior points per each epoch
num_epochs = 800
learning_rate=.002

# define approximating model
x_input = keras.layers.Input(shape=[1], name='x_input')
t_input = keras.layers.Input(shape=[1], name='t_input')
concat = keras.layers.Concatenate(axis=-1)([x_input, t_input])
prev_layer = concat
for i, units in enumerate(model_shape):
    if(i < len(model_shape) - 1):    
        prev_layer = keras.layers.Dense(units, activation='sigmoid', use_bias=True, trainable=True)(prev_layer)
    else: # final layer shouldn't have sigmoid activation
        prev_layer = keras.layers.Dense(units, use_bias=True, trainable=True)(prev_layer)
model = keras.Model(
    inputs=[x_input, t_input],
    outputs=prev_layer
)

@tf.function
def physics_guided_loss(x_interior, t_interior, t_initial, x_bound1, x_bound2):
    w_interior = model([x_interior, t_interior])
    w_initial = model([x_interior, t_initial])
    w_bound1 = model([x_bound1, t_interior])
    w_bound2 = model([x_bound2, t_interior])
    
    # wave layer enforcing differential equation
    wave_loss_tensor = wave_loss(w_interior, x_interior, t_interior, lam)
    # enforcing initial conditions
    initial_loss_tensor = initial_loss(w_initial, x_interior, t_initial)
    # enforcing boundary conditions
    boundary_loss_tensor = boundary_loss(w_bound1, w_bound2)
    #total loss
    loss = tf.reduce_mean(rhof * wave_loss_tensor + rho0 * initial_loss_tensor + rhob * boundary_loss_tensor)
    grads = tf.gradients(loss, model.weights)
    
    return [loss, grads]
#%%
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
# keep results for plotting
train_loss_results = []
train_accuracy_results = []

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    # create stochastic training points
    x_int = np.random.rand(Nf).astype(np.float32) * L
    t_int = np.random.rand(Nf).astype(np.float32) * T
    
    x_interior = x_int.reshape(-1, batch_size, 1)
    t_interior = t_int.reshape(-1, batch_size, 1)
    
    t_init = tf.zeros((batch_size, 1), dtype=tf.float32)
    x_bound1 = tf.zeros((batch_size, 1), dtype=tf.float32)
    x_bound2 = tf.constant(L, shape=(batch_size, 1), dtype=tf.float32)
    
    for x_int, t_int in zip(x_interior, t_interior):
        x_int = tf.constant(x_int)
        t_int = tf.constant(t_int)
        # Optimize the model
        loss_value, grads = physics_guided_loss(x_int, t_int, t_init, x_bound1, x_bound2)
        optimizer.apply_gradients(zip(grads, model.weights))
        
        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
    
    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    
    if(epoch % 1 == 0):
        print("Epoch {:03d}: Loss: {:.8f}".format(epoch, epoch_loss_avg.result()))
#%%
x_axis = np.linspace(0, L, num=500)
t_axis = np.linspace(0, T, num=500)

x_mesh, t_mesh = np.meshgrid(x_axis, t_axis)

# coords = np.vstack((x_mesh.reshape(-1), t_mesh.reshape(-1)))

x_in = np.reshape(x_mesh, [-1, 1])
t_in = np.reshape(t_mesh, [-1, 1])

w_pred = model.predict([x_in, t_in])
w_pred = w_pred.reshape(500, 500).T

fig = plt.figure(figsize=(7, 2))
pc = plt.pcolormesh(x_mesh, t_mesh, w_pred)
fig.colorbar(pc)
plt.xlabel(r'$t$ (ul)')
plt.ylabel(r'$x$ (ul)')
plt.savefig("physics-informed results v2.png", dpi=500)
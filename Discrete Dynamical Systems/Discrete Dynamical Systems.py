''' saving .gif doesn't work well in Jupyter'''
# works both on scalars and properly formatted numpy arrays
logistic = lambda x, l: l*x*(1-x)
# initialize arrays
import numpy as np

x_bounds = (0, 1)
lambda_bounds = (0, 4)
x_num = 500
lambda_resolution = 400
iterates = 256

lambda_points = np.linspace(lambda_bounds[0], lambda_bounds[1], num=lambda_resolution, endpoint=True)
x_start = np.linspace(x_bounds[0], x_bounds[1], num=x_num, endpoint=True)
x_, l_ = np.meshgrid(x_start, lambda_points)
# animate
import matplotlib.pyplot as plt
import matplotlib

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.set_xlim(lambda_bounds)
ax.set_ylim((x_bounds[0]-.15, x_bounds[1]+.15))
ax.set_xlabel(u"\u03BB")
ax.set_ylabel("x")

l, = ax.plot([],[], marker = '.', markersize=.25, linewidth=0, c='k')

def animate(i):
    global x_
    l.set_data(l_.reshape(-1), x_.reshape(-1))
    x_ = logistic(x_, l_)

ani = matplotlib.animation.FuncAnimation(fig, animate, frames=iterates)
ani.save("bifurcation evolution.gif", writer='ffmpeg', fps=10, dpi=150)
import kf_book.book_plots as book_plots
from kf_book.gh_internal import plot_g_h_results
import matplotlib.pylab as pylab
import pylab as plt
import time
import numpy as np

def g_h_filter(data, x0, dx, g, h, dt):
  """Performs g-h filter on 1 state variable with a fixed g and h.

    'data' contains the data(readings from the sensor) to be filtered.
    'x0' is the initial value for our state variable
    'dx' is the initial change rate for our state variable
    'g' is the g-h's g scale factor
    'h' is the g-h's h scale factor
    'dt' is the length of the time step
  """
  x = x0
  meas = []
  for z in data:
    # predict
    predict = x + dx * dt

    # update
    residual = z - predict
    x = predict + g * residual
    dx = dx + h * residual / dt

    meas.append(x)

  return np.array(meas)

%matplotlib inline
weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6,
           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]
plt.figure(figsize=(16, 9))
book_plots.plot_track([0, 11], [160, 172], label='Actual weight')
data = g_h_filter(data=weights, x0=160., dx=1., g=6./10, h=2./3, dt=1.)
plot_g_h_results(weights, data)

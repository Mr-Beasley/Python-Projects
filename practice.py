import numpy as np
import pylab as plt

X = np.linspace(0,5,100)
Y1 = X + 2*np.random.random(X.shape)
Y2 = X**2 + np.random.random(X.shape)

fig, ax = plt.subplots()
ax.plot(X,Y1,'o')
ax.plot(X,Y2,'x')
plt.show()
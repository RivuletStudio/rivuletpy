from rivuletpy.utils import backtrack
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pos = [200,200,200]
vec = [5, 1.2, 1.4]

X, Y, Z = backtrack.genrotmesh(vec, pos, 30)
fig = plt.figure()
ax = fig.gca(projection='3d')
print(X)
ax.plot_wireframe(X.reshape((61,61)), Y.reshape((61,61)), Z.reshape((61,61)))
plt.show()

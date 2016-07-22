from filtering.anisotropic import *
from rivuletpy.utils.io import * 
import matplotlib.pyplot as plt
from scipy import io as sio

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

# plot the gaussian kernel
nsig = 5
nmu = 5
kerlen = 101
kr = (kerlen - 1) / 2 
X, Y, Z = np.meshgrid(np.arange(-kr, kr+1),
	                  np.arange(-kr, kr+1), 
	                  np.arange(-kr, kr+1))
indstack = np.stack((X, Y, Z))
dist = np.linalg.norm(indstack, axis=0) 

plt.title('Gaussian')
for i in range(nsig):
	for j in range(nmu):
		mu = float(j*10)
		sigma = float(i+4)
		k = gkern3(dist, mu, sigma)
		ax = plt.subplot(nsig, nmu, (i) * nsig + (j + 1))
		ax.set_title('mu=%.2f, sigma=%.2f' % (mu, sigma))
		imgplot = plt.imshow(k[:, :, int((kerlen-1)/2)])
plt.colorbar()


plt.figure(2)
plt.title('Bi-Gaussian')
nsig = 5
nrho = 5
kerlen = 101
for i in range(nsig):
	for j in range(nrho):
		sigma = float(i * 10)+1
		rho = (j+1) * 0.1
		k = bgkern3(kerlen, 0, sigma, rho)
		ax = plt.subplot(nsig, nrho, (i) * nsig + (j + 1))
		ax.set_title('sigma=%.2f, rho=%.2f' % (sigma, rho))
		imgplot = plt.imshow(k[:, :, int((kerlen-1)/2) ])
plt.colorbar()
plt.show()


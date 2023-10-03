import time
import matplotlib.pyplot as plt
import numpy as np

dfile = "/mnt/md126/users/matin/dynamics_diffusion/scripts/../image_samples/sde/20231003-184612/samples_1x3x32x32.npz"
images = np.load(dfile)["arr_0"]
images = images.transpose((0, 2, 3, 1))
print(images.shape)
plt.ion()
plt.figure()
plt.imshow(images[0])
print("Press any key to continue...")
input()

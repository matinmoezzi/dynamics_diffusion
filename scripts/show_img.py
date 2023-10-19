import time
import matplotlib.pyplot as plt
import numpy as np

dfile = "/home/matin/data2/dynamics_diffusion/logs/ddpm/cifar10/gaussian/IUNet/sample_checkpoint_0_20231019-184411/logs/ddpm/cifar10/gaussian/IUNet/sample_checkpoint_0_20231019-184411/sample/samples_1x32x32x3.npz"
images = np.load(dfile)["arr_0"]
# images = images.transpose((0, 2, 3, 1)
print(images.shape)
plt.ion()
plt.figure()
plt.imshow(images[0])
print("Press any key to continue...")
input()

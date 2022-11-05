import numpy as np
# np.load("/media/giancos/Football/SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/BUTTA.npy")
original = np.load("data/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_ResNET_TF2_PCA512.npy")
imitation = np.load("results/features_test.npy")

print(original.shape)
print(imitation.shape)

# compare the two
print(np.allclose(original, imitation))


"""Create and save a range of centroids for varying number of abstract states"""

import numpy as np


def create_centroids(env_name, numz, norm=False):
    if env_name == "asterix":
        state_dim = 128
    elif env_name == "cartpole":
        state_dim = 4
    elif env_name == "sepsis":
        state_dim = 747
        norm = True
    if norm:
        centroids = np.random.randn(numz, state_dim)
    else:
        centroids = np.random.uniform(size=(numz, state_dim))
    np.save(centroids, f"../centroids_chkpt/{env_name}/unif_numclus{numz}.npy")


if __name__ == "__main__":
    numz_arr = [2, 4, 8, 16, 32, 64, 128]
    envs = ["asterix", "cartpole", "sepsis"]

    for env_name in envs:
        for numz in numz_arr:
            create_centroids(env_name, numz)

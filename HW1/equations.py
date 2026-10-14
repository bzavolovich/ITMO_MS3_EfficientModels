import numpy as np


def flops(S, B):
    return np.asarray(B) * (17714 * np.asarray(S)**2 + 313700)

def memory(S, B):
    return 4161296 + 68 * np.asarray(B) * np.asarray(S)**2

def bytes_moved(S, B):
    return 4 * (1040324 + 95 * np.asarray(B) * np.asarray(S)**2 + 2148 * np.asarray(B))

def latency(S, B, theta):
    tau, compute, bandwidth = theta
    return tau + np.maximum(flops(S, B) / compute, bytes_moved(S, B) / bandwidth)

def energy(S, B, theta_energy):
    return theta_energy[3] * latency(S, B, theta_energy[:3])

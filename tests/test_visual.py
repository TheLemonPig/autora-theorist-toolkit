import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    n_samples = 50
    x1 = np.linspace(-8, 8, n_samples).reshape(-1, 1)
    x2 = np.linspace(-8, 8, n_samples).reshape(-1, 1)
    x1, x2 = np.meshgrid(x1, x2)
    X = np.hstack((x1, x2))
    w1, w2 = np.sin(x1) ** 2, np.cos(x2) ** 3 + np.sin(x1) ** 2
    c0, c1, c2 = 0.4, 0.8, 1.2
    y = c0 + c1 * x1 + c2 * w2
    z = 1 / (1 + np.exp(-y))
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.scatter(x1, x2, z)
    fig.tight_layout()
    plt.show()

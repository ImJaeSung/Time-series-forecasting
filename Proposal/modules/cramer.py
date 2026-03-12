import torch
import math

def phi_D_approx(s, D, eps=1e-12):
    """
    Asymptotic approximation of \phi_D(s)
    \phi_D(s) ≈ (1 + 4s / (2D - 3))^{-1/2},  valid for D ≥ 20
    """
    return torch.pow(1.0 + 4.0 * s / (2.0 * D - 3.0 + eps), -0.5)


def pairwise_sq_dist(X, Y):
    """
    Compute pairwise squared Euclidean distances
    X: (B, D)
    Y: (B, D)
    returns: (B, B)
    """
    X_norm = (X ** 2).sum(dim=1, keepdim=True)
    Y_norm = (Y ** 2).sum(dim=1, keepdim=True)
    return X_norm + Y_norm.T - 2.0 * X @ Y.T


def cramer_wold_distance(X, Y, gamma=None):
    """
    Cramer Wold distance d^2_cw(X, Y)

    X, Y: (B, D) torch tensors
    gamma: smoothing parameter (if None: Silverman's rule)

    returns: scalar tensor
    """
    assert X.shape == Y.shape
    B, D = X.shape
    device = X.device
    dtype = X.dtype

    # Silverman's rule of thumb
    if gamma is None:
        gamma = (4.0 / (3.0 * B)) ** (2.0 / 5.0)

    # pairwise distances
    XX = pairwise_sq_dist(X, X)
    YY = pairwise_sq_dist(Y, Y)
    XY = pairwise_sq_dist(X, Y)

    # argument of \phi_D
    s_xx = XX / (4.0 * gamma)
    s_yy = YY / (4.0 * gamma)
    s_xy = XY / (4.0 * gamma)

    # \phi_D
    K_xx = phi_D_approx(s_xx, D)
    K_yy = phi_D_approx(s_yy, D)
    K_xy = phi_D_approx(s_xy, D)

    # CW distance
    const = 1.0 / (2.0 * B * B * math.sqrt(math.pi * gamma))
    loss = const * (
        K_xx.sum()
        + K_yy.sum()
        - 2.0 * K_xy.sum()
    )

    return loss
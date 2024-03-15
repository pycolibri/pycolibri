import torch


def sensingH(hyperimg, gdmd):
    [L, M, N] = hyperimg.shape
    S = gdmd.shape[0]
    y = torch.zeros(S, M, N + L - 1)

    y_noshift = hyperimg[:, None] * gdmd
    for k in range(L):
        y[..., k:N + k] = y[..., k:N + k] + y_noshift[k]

    return y

def sensingHt(b, gdmd):
    [_, M, N] = gdmd.shape
    L = b.shape[-1] - N + 1

    y = torch.zeros(L, M, N)
    for k in range(L):
        y[k] = y[k] + torch.sum(b[..., k:N + k] * gdmd, dim=0)

    return y
    

def computeP(f, L):
    [S, M, N] = f.shape
    P = torch.zeros(S, S, M, N + L - 1)
    for t in range(S):
        for s in range(t, S):
            # This loop corresponds to the formula (5), i.e. sum_{l=1}^L S^Z_{l-1}(f_t).*S^Z_{l-1}(f_s)
            for l in range(L):
                P[t, s, :, l:N + l] = P[t, s, :, l:N + l] + f[t] * f[s]

            # Here we use the fact that the matrix is "self-adjoint", so we do not need
            # to compute lower diagonal images
            P[s, t] = P[t, s]

    return P


def computeQ(f, L):
    [_, M, N] = f.shape
    Q = torch.zeros(L, L, M, N)
    # we first evaluate the first row of Q's, so k=1 below
    for l in range(L):
        Q[0, l, :, l:N] = torch.sum(f[..., :N - l] * f[..., l:N], dim=0)

    # We now compute remaining indices using the symmetry remarks at
    # the endo of section 3.4
    for k in range(L):
        for l in range(k, L):
            # This line uses the Teoplitz structure to get the upper
            # diagonal elements
            Q[k, l] = Q[0, l - k]
            # And this line the formula Q_{l,k}=S_{k-l}[Q_{k,l}],
            # to get the lower diagonal elements
            Q[l, k, :, :N - l + k] = Q[k, l, :, l - k:N]

    return Q


def IMVM(P, b):
    [S, M, N] = b.shape
    y = torch.sum(P * b[None], dim=1)

    return y

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator


def lanczos(
    operator,
    size,
    num_lanczos_vectors,
    use_gpu=False,
):
    """
    Parameters
    -------------
    operator: ModelHessianOperator.
    num_lanczos_vectors : int
        number of lanczos vectors to compute.
    use_gpu: bool
        if true, use cuda tensors.

    Returns
    ----------------
    T: a nv x m x m tensor, T[i, :, :] is the ith symmetric tridiagonal matrix
    V: a n x m x nv tensor, V[:, :, i] is the ith matrix with orthogonal rows
    """

    shape = (size, size)

    def _scipy_apply(x):
        x = torch.from_numpy(x)
        if use_gpu:
            x = x.cuda()
        return operator.apply(x.float()).cpu().numpy()

    scipy_op = ScipyLinearOperator(shape, _scipy_apply)

    from hessian_eigenthings.slq_upd import _lanczos_m_upd
    # vec = np.random.random(size=(shape[0], 1)).astype(float)
    # vec = np.random.randn(shape[0], 1)
    T, V = _lanczos_m_upd(A=scipy_op, m=num_lanczos_vectors, matrix_shape=shape, SV=None)
    return T, V
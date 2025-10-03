import math

import torch
from torch import linalg as LA


@torch.no_grad()
def _matrix_sqrt_psd(Sigma: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # symmetric PSD matrix square root via eigen decomposition
    Sigma = 0.5 * (Sigma + Sigma.t())
    d = Sigma.shape[0]
    Sigma = Sigma + eps * \
        torch.eye(d, dtype=Sigma.dtype, device=Sigma.device)
    evals, evecs = torch.linalg.eigh(Sigma)  # ascending
    evals = torch.clamp(evals, min=0.0).sqrt()
    return (evecs * evals.unsqueeze(0)) @ evecs.t()


@torch.no_grad()
def cur_deim_gpu(W: torch.Tensor,
                 r: int,
                 use_lowrank: bool = True,
                 oversample: int = 20,
                 niter: int = 2) -> tuple[list[int], list[int]]:
    """
    DEIM-CUR 선택 알고리즘 (GPU friendly)

    Args:
        W  : (m, n) weight or importance matrix (GPU/CPU 모두 가능)
        r  : target rank  (r << min(m, n))
        use_lowrank : True → torch.linalg.svd_lowrank (권장)
                      False → thin-SVD (full_matrices=False)
        oversample   : low-rank SVD 시 여유차원 (q = r + oversample)
        niter        : randomized SVD power iteration 횟수

    Returns:
        irow, icol : 선택된 row / col 인덱스를 Python 리스트로 반환
    """

    # --- ① SVD (thin or low-rank) ------------------------------------------
    if use_lowrank:
        # randomized / block Lanczos SVD (GPU 지원)
        # print(f"DEIM-CUR: SVD lowrank ({W.device})")
        qmax = min(W.shape[0], W.shape[1])
        q = min(r + oversample, qmax)
        U, S, V = torch.svd_lowrank(W, q=q, niter=niter)
        U, V = U[:, :r], V[:, :r]
    else:
        # thin-SVD
        # print(f"DEIM-CUR: SVD ({W.device})")
        U, S, Vh = LA.svd(W, full_matrices=False)
        U, V = U[:, :r], Vh.T[:, :r]

    # --- ② DEIM 선택 (GPU 텐서 연산, no .item(), no CPU sync) --------------
    m, n = W.shape
    irow = torch.empty(r, dtype=torch.long, device=W.device)
    icol = torch.empty(r, dtype=torch.long, device=W.device)
    mask_r = torch.zeros(m, dtype=torch.bool, device=W.device)
    mask_c = torch.zeros(n, dtype=torch.bool, device=W.device)

    for k in range(r):
        # 가장 큰 절대값(중복 제외)
        u_vec = torch.where(mask_r, torch.zeros_like(U[:, k]), U[:, k].abs())
        v_vec = torch.where(mask_c, torch.zeros_like(V[:, k]), V[:, k].abs())
        row_k = torch.argmax(u_vec)
        col_k = torch.argmax(v_vec)

        irow[k] = row_k
        icol[k] = col_k
        mask_r[row_k] = True
        mask_c[col_k] = True

        # --- ③ rank-1 업데이트 (폐쇄형) : pinverse(1×k) = vec / ||vec||² -------
        if k + 1 < r:
            alpha_r = U[row_k, :k+1]            # (k+1,)
            alpha_c = V[col_k, :k+1]            # (k+1,)

            denom_r = (alpha_r @ alpha_r).clamp_min(1e-12)
            denom_c = (alpha_c @ alpha_c).clamp_min(1e-12)
            U[:, k+1:] -= (U[:, :k+1] @ alpha_r.unsqueeze(1)) / denom_r
            V[:, k+1:] -= (V[:, :k+1] @ alpha_c.unsqueeze(1)) / denom_c

    return irow.tolist(), icol.tolist()


def select_rows_and_columns(
    W, A, num_rows, num_cols,
    aux_mode: str = 'wanda',
    cur_mode: str = 'deim',
):
    """
    Select rows and columns from matrix W based on WANDA metrics computed using activation norms.

    - W: weight matrix of shape [out_features, in_features]
    - A:
        * 'wanda'    : tensor of shape [in_features]          (sqrt(E[x^2]))
        * 'cov_fast' : tensor of shape [in_features]          (sqrt(E[x^2]))
        * 'cov'      : tensor of shape [in_features, in_features] (Cov[x])
    - aux_mode: 'wanda' | 'cov_fast' | 'cov'

    Selection is performed on:
        'wanda'    → S = |W| * act
        'cov_fast' → M = W * scale
        'cov'      → M = W @ Cov^{1/2}
    """

    # if num_cols != num_rows:
    #     raise ValueError("Not a square matrix.")
    m, n = W.shape
    r = min(num_rows, num_cols, m, n)

    # Fast-out
    if cur_mode == 'random':
        k_rows = min(num_rows, m)
        k_cols = min(num_cols, n)
        row_indices = torch.randperm(m, device=W.device)[:k_rows].tolist()
        col_indices = torch.randperm(n, device=W.device)[:k_cols].tolist()
        return row_indices, col_indices

    # AUX

    if aux_mode == 'wanda':
        act = A.view(1, -1).to(W.device, dtype=W.dtype)
        S = W.abs() * act  # Hadamard multiplication

    elif aux_mode == 'weight':
        S = W.abs()

    elif aux_mode == 'cov_fast':
        scale = A.view(1, -1).to(W.device, dtype=W.dtype)
        S = W * scale

    elif aux_mode == 'cov':
        Sigma = A.to(W.device, dtype=W.dtype)
        D = _matrix_sqrt_psd(Sigma)
        S = W @ D

    # CUR

    if cur_mode == 'deim':
        row_indices, col_indices = cur_deim_gpu(
            S, r,
            use_lowrank=True
        )  # GPU
        return row_indices, col_indices

    elif cur_mode == 'deim_full':
        row_indices, col_indices = cur_deim_gpu(
            S, r,
            use_lowrank=False
        )  # GPU
        return row_indices, col_indices

    elif cur_mode == 'magnitude':
        # Magnitude (Prob) with stability guards

        # S는 위 metric_mode 분기에서 이미 정의됨
        frobenius_norm = torch.norm(S, p='fro')
        col_norms = torch.norm(S, p=2, dim=0)  # (n,)
        row_norms = torch.norm(S, p=2, dim=1)  # (m,)

        k_cols = min(num_cols, col_norms.numel())
        k_rows = min(num_rows, row_norms.numel())

        # Degenerate cases → deterministic top-k fallback
        cond_bad = (
            (not torch.isfinite(frobenius_norm)) or (frobenius_norm <= 0) or
            (not torch.isfinite(col_norms).all()) or (not torch.isfinite(row_norms).all()) or
            (col_norms.sum() <= 0) or (row_norms.sum() <= 0)
        )
        if cond_bad:
            col_indices = torch.topk(col_norms, k_cols, largest=True).indices
            row_indices = torch.topk(row_norms, k_rows, largest=True).indices
            return row_indices.tolist(), col_indices.tolist()

        # Probabilistic sampling without replacement (normalized)
        col_prob = (col_norms / col_norms.sum()).clamp_min(0)
        row_prob = (row_norms / row_norms.sum()).clamp_min(0)

        col_indices = torch.multinomial(
            col_prob, num_samples=k_cols, replacement=False)
        row_indices = torch.multinomial(
            row_prob, num_samples=k_rows, replacement=False)
        return row_indices.tolist(), col_indices.tolist()

        # # Magnitude
        # # Sum over out_features
        # col_importance = S.sum(dim=0)
        # num_cols = min(num_cols, col_importance.size(0))
        # col_indices = torch.topk(col_importance, num_cols, largest=True)[1]
        # # Sum over in_features
        # row_importance = S.sum(dim=1)
        # num_rows = min(num_rows, row_importance.size(0))
        # row_indices = torch.topk(row_importance, num_rows, largest=True)[1]
        # # return
        # return row_indices.tolist(), col_indices.tolist()


def cur_decomposition(W, A, num_rows, num_cols, aux_mode: str = 'wanda', cur_mode: str = 'deim', use_float64: bool = True):
    """
    Perform CUR decomposition on matrix W using WANDA metrics.
    """

    orig_dtype = W.dtype
    if use_float64 and orig_dtype != torch.float64:
        W = W.to(torch.float64)
        if A is not None:
            A = A.to(torch.float64)

    row_indices, col_indices = select_rows_and_columns(
        W, A, num_rows, num_cols,
        aux_mode=aux_mode, cur_mode=cur_mode)

    C = W[:, col_indices]
    R = W[row_indices, :]

    rc = 1e-12 if orig_dtype == torch.float64 else 1e-6
    # rc = None

    if aux_mode == 'wanda':
        U = (
            torch.linalg.pinv(C, rcond=rc)
            @ W
            @ torch.linalg.pinv(R, rcond=rc)
        )

    elif aux_mode == 'cov_fast':
        # A: (n,) = sqrt(E[x^2]) expected
        if A.dim() != 1 or A.numel() != W.shape[1]:
            raise ValueError(
                f"[cov_fast] scale vector shape mismatch: expected ({W.shape[1]},) got {tuple(A.shape)}")
        Dvec = A.to(W.device, dtype=W.dtype)           # (n,)
        WD = W * Dvec.view(1, -1)                      # (m,n)
        RD = R * Dvec.view(1, -1)                      # (r,n)
        U = (
            torch.linalg.pinv(C, rcond=rc)
            @ WD
            @ torch.linalg.pinv(RD, rcond=rc)
        )

    elif aux_mode == 'cov':
        # A: (n,n) = Cov[x] expected
        if A.dim() != 2 or A.shape[0] != W.shape[1] or A.shape[1] != W.shape[1]:
            raise ValueError(
                f"[cov] covariance shape mismatch: expected ({W.shape[1]},{W.shape[1]}) got {tuple(A.shape)}")
        Sigma = A.to(W.device, dtype=W.dtype)          # (n,n)
        D = _matrix_sqrt_psd(Sigma)                    # Cov^{1/2}
        WD = W @ D                                     # (m,n)
        RD = R @ D                                     # (r,n)
        U = (
            torch.linalg.pinv(C, rcond=rc)
            @ WD
            @ torch.linalg.pinv(RD, rcond=rc)
        )

    else:
        # raise ValueError(f"Unknown aux_mode: {aux_mode}")
        U = (
            torch.linalg.pinv(C, rcond=rc)
            @ W
            @ torch.linalg.pinv(R, rcond=rc)
        )

    if use_float64 and orig_dtype != torch.float64:
        C = C.to(orig_dtype)
        R = R.to(orig_dtype)
        U = U.to(orig_dtype)

    return C, U, R, row_indices, col_indices


@torch.no_grad()
def energy_rank(W: torch.Tensor,
                A: torch.Tensor,
                aux_mode: str,
                energy: float = 0.98,
                use_lowrank: bool = True,
                niter: int = 2) -> int:
    """
    Decide rank based on retained energy ratio on M (or S).

    aux_mode:
      - 'wanda'    : S = |W| * act, act: (n,) ≈ sqrt(E[x^2])
      - 'cov_fast' : M = W * scale,   scale: (n,) = sqrt(E[x^2])  (sign-preserving)
      - 'cov'      : M = W @ Cov^{1/2}, Cov: (n,n)
    """
    m, n = W.shape

    # Build selection/weighted matrix
    if aux_mode == 'wanda':
        if A.dim() != 1 or A.numel() != n:
            raise ValueError(
                f"[wanda] aux vector shape mismatch: expected ({n},) got {tuple(A.shape)}")
        act = A.view(1, -1).to(W.device, dtype=W.dtype)
        M = W.abs() * act  # S = |W| * act

    elif aux_mode == 'weight':
        M = W.abs()

    elif aux_mode == 'cov_fast':
        if A.dim() != 1 or A.numel() != n:
            raise ValueError(
                f"[cov_fast] scale vector shape mismatch: expected ({n},) got {tuple(A.shape)}")
        scale = A.view(1, -1).to(W.device, dtype=W.dtype)
        M = W * scale  # M = W * sqrt(E[x^2])

    elif aux_mode == 'cov':
        if A.dim() != 2 or A.shape[0] != n or A.shape[1] != n:
            raise ValueError(
                f"[cov] covariance shape mismatch: expected ({n},{n}) got {tuple(A.shape)}")
        Sigma = A.to(W.device, dtype=W.dtype)
        D = _matrix_sqrt_psd(Sigma)  # Cov^{1/2}
        M = W @ D  # M = W @ Cov^{1/2}

    else:
        raise ValueError(f"Unknown aux_mode: {aux_mode}")

    # Energy-based rank from singular values of M
    if use_lowrank:
        q = min(
            max(256, int(min(m, n) * 0.25)),  # TODO: max 25% or 256
            min(m, n)
        )
        _, sv, _ = torch.svd_lowrank(M.float(), q=q, niter=niter)
    else:
        # SLOW
        # sv = torch.linalg.svdvals(M.float())  # descending
        _, sv, _ = torch.linalg.svd(M.float(), full_matrices=False)

    if sv.numel() == 0:
        r = 1
    else:
        e = sv.square()
        total = e.sum()
        if total <= 0 or not torch.isfinite(total):
            r = 1
        else:
            cume = torch.cumsum(e, dim=0) / total
            target = float(energy)
            # keep target within (0,1)
            if not (0.0 < target < 1.0):
                target = max(1e-6, min(target, 0.999999))
            r = int(torch.searchsorted(cume, torch.tensor(
                target, device=cume.device)).item()) + 1

    # Round up to the nearest multiple of 128
    # This can improve hardware efficiency (e.g., tensor cores).
    # We round up to preserve at least the energy target.
    if r > 0:
        r = ((r + 127) // 128) * 128

    # Guards
    r = max(1, min(r, min(m, n)))
    return r


def calculate_rank(m, n):
    """
    Calculate the rank for CUR decomposition based on matrix dimensions m and n.
    """
    try:
        r = int((math.sqrt(m**2 + 6 * m * n + n**2) - (m + n)) / 2)
    except ValueError:
        # This can happen if the term inside sqrt is negative, though unlikely with m,n > 0
        r = min(m, n)

    # Round down to the nearest multiple of 128
    # We round down to stay below the parameter breakeven point.
    if r > 0:
        r = (r // 128) * 128

    # Guards
    r = max(1, min(r, min(m, n)))
    return r


def apply_cur_to_matrix(weight, aux_info,
                        max_rank=None, min_rank=None,
                        aux_mode: str = 'wanda', cur_mode: str = 'deim',
                        energy: float | None = None):  # 0.98
    """
    Apply CUR decomposition to a single weight matrix using WANDA metrics.

    Args:
        weight          : (m,n) matrix W
        aux_info        : (n,) for 'wanda'/'cov_fast', (n,n) for 'cov'
        min_rank        : optional rank floor
        aux_mode        : 'wanda' | 'cov_fast' | 'cov'
        energy          : retained energy ratio for rank selection (None → use size heuristic)
    """
    m, n = weight.shape
    if energy is not None:
        rank = energy_rank(
            weight,
            aux_info,
            aux_mode=aux_mode,
            energy=energy,
            use_lowrank=False,
        )
        upper_bound_rank = calculate_rank(m, n)
        if rank > upper_bound_rank:
            raise ValueError("No compression.")
    else:
        rank = calculate_rank(m, n)

    if max_rank:
        rank = min(rank, int(max_rank))
    if min_rank:
        rank = max(rank, int(min_rank))

    C, U, R, row_indices, col_indices = cur_decomposition(
        weight, aux_info, num_rows=rank, num_cols=rank,
        aux_mode=aux_mode, cur_mode=cur_mode)
    return C, U, R, rank, row_indices, col_indices

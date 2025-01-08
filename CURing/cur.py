import math

import torch


def cur_deim(A, r):
    """
    - A: input matrix (dataset as a torch tensor)
    - r: desired rank
    """

    m, n = A.shape
    if r > m or r > n:
        raise ValueError(
            "Desired rank is greater than the dimensions of the input matrix.")

    # Perform SVD using PyTorch
    u, s, vh = torch.linalg.svd(A, full_matrices=True)
    U = u[:, :r]
    V = vh.T[:, :r]

    irow = []
    icol = []
    for i in range(r):
        # Remove redundancy
        # row_candidates = torch.topk(torch.abs(U[:, i]), k=U.shape[0])[1]
        # col_candidates = torch.topk(torch.abs(V[:, i]), k=V.shape[0])[1]
        row_candidates = torch.sort(torch.abs(U[:, i]), descending=True)[1]
        col_candidates = torch.sort(torch.abs(V[:, i]), descending=True)[1]
        row_i = next(row.item()
                     for row in row_candidates if row.item() not in irow)
        col_i = next(col.item()
                     for col in col_candidates if col.item() not in icol)

        # Update U and V
        if i + 1 < r:
            U[:, i + 1:] -= U[:, :i + 1] @ torch.pinverse(
                U[row_i, :i + 1].unsqueeze(0)) @ U[row_i, i + 1:].unsqueeze(0)
            V[:, i + 1:] -= V[:, :i + 1] @ torch.pinverse(
                V[col_i, :i + 1].unsqueeze(0)) @ V[col_i, i + 1:].unsqueeze(0)

        irow.append(row_i)
        icol.append(col_i)

    return irow, icol


def select_rows_and_columns(A, activation_norm, num_rows, num_cols):
    """
    Select rows and columns from matrix A based on WANDA metrics computed using activation norms.

    - A: weight matrix of shape [out_features, in_features]
    - activation_norm: tensor of shape [in_features]
    """

    activation_norm = activation_norm.view(1, -1)


    # ========================

    ########
    # TEST #
    ########


    # WANDA
    W_metric = A.abs() * activation_norm  # Hadamard multiplication

    # # Weight
    # W_metric = A.abs()


    # # Magnitude
    # # Sum over out_features
    # col_importance = W_metric.sum(dim=0)
    # num_cols = min(num_cols, col_importance.size(0))
    # col_indices = torch.topk(col_importance, num_cols, largest=True)[1]
    # # Sum over in_features
    # row_importance = W_metric.sum(dim=1)
    # num_rows = min(num_rows, row_importance.size(0))
    # row_indices = torch.topk(row_importance, num_rows, largest=True)[1]
    # # return
    # return row_indices.tolist(), col_indices.tolist()

    # # Magnitude (Prob)
    # frobenius_norm = torch.norm(W_metric, p='fro')
    # # Sum over out_features
    # col_norms = torch.norm(W_metric, p=2, dim=0)
    # col_importance = col_norms / frobenius_norm
    # num_cols = min(num_cols, col_importance.size(0))
    # col_indices = torch.multinomial(col_importance, num_samples=num_cols, replacement=False)
    # # Sum over in_features
    # row_norms = torch.norm(W_metric, p=2, dim=1)
    # row_importance = row_norms / frobenius_norm
    # num_rows = min(num_rows, row_importance.size(0))
    # row_indices = torch.multinomial(row_importance, num_samples=num_rows, replacement=False)
    # # return
    # return row_indices.tolist(), col_indices.tolist()

    # DEIM
    if num_cols != num_rows:
        raise ValueError("Not a square matrix.")
    row_indices, col_indices = cur_deim(W_metric, num_cols)
    # return
    return row_indices, col_indices


    # # RANDOM
    # m, n = A.shape
    # row_indices = torch.randperm(m)[:num_rows].tolist()
    # col_indices = torch.randperm(n)[:num_cols].tolist()
    # # return
    # return row_indices, col_indices
    
    # ========================


def cur_decomposition(A, activation_norm, num_rows, num_cols):
    """
    Perform CUR decomposition on matrix A using WANDA metrics.
    """
    row_indices, col_indices = select_rows_and_columns(
        A, activation_norm, num_rows, num_cols)

    C = A[:, col_indices]
    R = A[row_indices, :]
    U = torch.pinverse(C) @ A @ torch.pinverse(R)

    return C, U, R, row_indices, col_indices


def calculate_rank(m, n, max_rank=256):
    """
    Calculate the rank for CUR decomposition based on matrix dimensions m and n.
    """
    return min(int((math.sqrt(m ** 2 + 6 * m * n + n ** 2) - (m + n)) / 2), max_rank)


def apply_cur_to_matrix(weight, activation_norm, max_rank):
    """
    Apply CUR decomposition to a single weight matrix using WANDA metrics.
    """
    m, n = weight.size()
    rank = calculate_rank(m, n, max_rank=max_rank)
    C, U, R, row_indices, col_indices = cur_decomposition(
        weight, activation_norm, num_rows=rank, num_cols=rank)
    return C, U, R, rank, row_indices, col_indices

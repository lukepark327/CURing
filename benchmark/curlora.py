import numpy as np
from torch import nn
import torch


def compute_selection_probabilities(A):
    column_norms_squared = torch.sum(A**2, axis=0)
    row_norms_squared = torch.sum(A**2, axis=1)
    total_sum_squares = torch.sum(column_norms_squared)
    column_probs = column_norms_squared / total_sum_squares
    row_probs = row_norms_squared / total_sum_squares
    return column_probs, row_probs


def select_indices_with_replacement(probs, k):
    #inverted_P = 1 / (probs + 0.001)
    inverted_P = (1 / (probs + 0.001)).float()

    # Normalize the inverted probabilities
    probs = inverted_P / inverted_P.sum()

    probs = probs.cpu().numpy()

    return np.random.choice(len(probs), size=k, replace=True, p=probs)


def adjust_duplicates(selected_indices, A, axis):
    unique_indices, counts = np.unique(selected_indices, return_counts=True)
    adjusted_matrix = A[:, unique_indices] if axis == 1 else A[unique_indices, :]
    
    for idx, count in enumerate(counts):
        if count > 1:
            scaling_factor = np.sqrt(count)
            if axis == 1:
                adjusted_matrix[:, idx] *= scaling_factor
            else:
                adjusted_matrix[idx, :] *= scaling_factor
    
    return adjusted_matrix, unique_indices


def cur_decomposition(A, c):
    r = c
    column_probs, row_probs = compute_selection_probabilities(A)
    selected_columns = select_indices_with_replacement(column_probs, c)
    selected_rows = select_indices_with_replacement(row_probs, r)
    
    C = A[:, selected_columns]
    R = A[selected_rows, :]
    
    U = torch.empty(C.shape[1], R.shape[0])
    U = torch.zeros_like(U)
    
    return C, U, R


class CURLoRAModule(nn.Module):
    def __init__(self, W, rank):
        super(CURLoRAModule, self).__init__()
        C, U, R = cur_decomposition(W, rank)
        self.register_buffer('C', C)
        self.register_buffer('R', R)
        self.U = nn.Parameter(U)  # U is trainable

    def forward(self, x):
        W_approx = torch.matmul(torch.matmul(self.C, self.U), self.R)
        x = x.matmul(W_approx.t())
        return x


class CURLoRALinear(nn.Module):
    def __init__(self, weight, bias=None, rank=256, alpha=1):
        super(CURLoRALinear, self).__init__()
        self.weight = weight
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None
        self.rank = rank
        self.alpha = alpha

        # CURLoRA
        self.curlora_modules = CURLoRAModule(self.weight, self.rank)
                
    def forward(self, x):
        x_0 = x.matmul(self.weight.t()) 
        x_adapted = self.curlora_modules(x)
        x = x_0 + (self.alpha * x_adapted)
        if self.bias is not None:
            x += self.bias
        return x

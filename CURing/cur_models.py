import torch
import torch.nn as nn


# CURLinear


class CURLinear(nn.Module):
    def __init__(self, C, U, R, bias=None, row_indices=None, col_indices=None):
        super(CURLinear, self).__init__()
        self.register_buffer('C', C)
        self.register_buffer('R', R)
        self.U = nn.Parameter(U)  # U is trainable
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

        # Store indices for interpretability
        self.row_indices = row_indices
        self.col_indices = col_indices

        # Placeholders for accumulated activations
        self.capture_activations = False
        self.activation_R_accum = None
        self.activation_C_accum = None
        self.nsamples = 0

    def forward(self, x):
        # y = ((x @ R.T) @ U.T) @ C.T
        out_R = x.matmul(self.R.t())  # Shape: (batch_size, seq_length, rank)
        # Shape: (batch_size, seq_length, rank)
        out_U = out_R.matmul(self.U.t())
        # Shape: (batch_size, seq_length, output_dim)
        out_C = out_U.matmul(self.C.t())

        if self.capture_activations:
            # Accumulate activation_R
            activation_R = out_R.detach()
            # Sum over batch and sequence dimensions
            activation_R_sum = activation_R.sum(dim=(0, 1))
            if self.activation_R_accum is None:
                self.activation_R_accum = activation_R_sum
            else:
                self.activation_R_accum += activation_R_sum

            # Accumulate activation_C
            activation_C = out_U.detach()  # Collect from out_U
            # Sum over batch and sequence dimensions
            activation_C_sum = activation_C.sum(dim=(0, 1))
            if self.activation_C_accum is None:
                self.activation_C_accum = activation_C_sum
            else:
                self.activation_C_accum += activation_C_sum

            # Update nsamples (total positions)
            total_positions = activation_R.size(
                0) * activation_R.size(1)  # batch_size * seq_length
            self.nsamples += total_positions

        if self.bias is not None:
            out_C += self.bias
        return out_C

    def reset_activations(self):
        self.capture_activations = False
        self.activation_R_accum = None
        self.activation_C_accum = None
        self.nsamples = 0

    def activate_capture(self):
        self.capture_activations = True

    def deactivate_capture(self):
        self.capture_activations = False


def activate_capture_for_all_CURLinear_modules(model):
    for module in model.modules():
        if isinstance(module, CURLinear):
            module.activate_capture()


def deactivate_capture_for_all_CURLinear_modules(model):
    for module in model.modules():
        if isinstance(module, CURLinear):
            module.deactivate_capture()


def reset_activations_for_all_CURLinear_modules(model):
    for module in model.modules():
        if isinstance(module, CURLinear):
            module.reset_activations()


def rebuild_model_with_W(model):
    """
    Rebuilds the model by replacing CURLinear modules with Linear modules with weight W = C @ U @ R.
    """
    for name, module in model.named_modules():
        if isinstance(module, CURLinear):
            # Reconstruct W = C @ U @ R
            C = module.C
            U = module.U
            R = module.R
            W = C @ U @ R
            bias = module.bias
            # Create a new Linear module
            in_features = R.size(1)  # Original input features
            out_features = C.size(0)  # Original output features
            linear = nn.Linear(
                in_features=in_features, out_features=out_features, bias=(bias is not None))
            linear.weight.data = W
            if bias is not None:
                linear.bias.data = bias
            # Replace the CURLinear module with the new Linear module
            # Navigate to the parent module
            parent_module = model
            name_parts = name.split('.')
            for part in name_parts[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, name_parts[-1], linear)
    return model


# WANDA


class WandaWrappedModule:
    def __init__(self, module, device='cuda'):
        self.module = module
        self.scaler_row = None
        self.nsamples = 0
        self.device = device

    def add_batch(self, module, inp, out):
        # module # input # output

        activation = inp[0].detach()
        # If batch
        if activation.dim() > 2:
            activation = activation.view(-1, activation.size(-1))
        activation = activation.to(self.device)

        if self.scaler_row is None:
            self.scaler_row = torch.zeros(activation.size(1), device=self.device)
        self.scaler_row += activation.pow(2).sum(dim=0)  # L2 norm

        batch_size = activation.size(0)
        self.nsamples += batch_size

    def register_hook(self):
        self.handle = self.module.register_forward_hook(self.add_batch)

    def remove_hook(self):
        self.handle.remove()

    def get_activation_norm(self):
        # L2 norm
        activation_norm = torch.sqrt(
            self.scaler_row / self.nsamples  # Average
        )
        return activation_norm


# Activation
# Hooks to collect activations (for interpretability)


class ActivationAccumulator:
    def __init__(self, module):
        self.module = module
        self.input_activation_accum = None
        self.output_activation_accum = None
        self.nsamples = 0
        self.capture_activations = False

    def add_batch(self, module, inp, out):
        if not self.capture_activations:
            return

        # Collect activations from module inputs
        # Shape: (batch_size, seq_length, input_dim)
        activation_inp = inp[0].detach()
        # Flatten to (total_positions, input_dim)
        activation_inp = activation_inp.view(-1, activation_inp.size(-1))
        activation_inp_sum = activation_inp.sum(
            dim=0)  # Sum over all positions
        if self.input_activation_accum is None:
            self.input_activation_accum = activation_inp_sum
        else:
            self.input_activation_accum += activation_inp_sum

        # Collect activations from module outputs
        activation_out = out.detach()  # Shape: (batch_size, seq_length, output_dim)
        activation_out = activation_out.view(-1, activation_out.size(-1))
        activation_out_sum = activation_out.sum(dim=0)
        if self.output_activation_accum is None:
            self.output_activation_accum = activation_out_sum
        else:
            self.output_activation_accum += activation_out_sum

        # Update nsamples
        self.nsamples += activation_inp.size(0)  # Total number of positions

    def register_hook(self):
        self.handle = self.module.register_forward_hook(self.add_batch)

    def remove_hook(self):
        self.handle.remove()

    def get_mean_input_activation(self):
        if self.input_activation_accum is not None and self.nsamples > 0:
            return self.input_activation_accum / self.nsamples
        else:
            return None

    def get_mean_output_activation(self):
        if self.output_activation_accum is not None and self.nsamples > 0:
            return self.output_activation_accum / self.nsamples
        else:
            return None

    def reset(self):
        self.input_activation_accum = None
        self.output_activation_accum = None
        self.nsamples = 0
        self.capture_activations = False

    def activate_capture(self):
        self.capture_activations = True

    def deactivate_capture(self):
        self.capture_activations = False


def activate_capture_for_all_ActivationAccumulator(wrapped_modules):
    for wrapped_module in wrapped_modules.values():
        wrapped_module.activate_capture()

def deactivate_capture_for_all_ActivationAccumulator(wrapped_modules):
    for wrapped_module in wrapped_modules.values():
        wrapped_module.deactivate_capture()

def reset_activations_for_all_ActivationAccumulator(wrapped_modules):
    for wrapped_module in wrapped_modules.values():
        wrapped_module.reset()

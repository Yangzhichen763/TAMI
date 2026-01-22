import torch
import torch.nn as nn


def kernel_smoothing(x, bandwidth=1.0, last_only=False):
    """Non-parametric kernel smoothing for autoregression.
    Args:
        x: Input tensor of shape (B, N, C).
        bandwidth: Smoothing parameter (higher = smoother).
    Returns:
        Predicted (B, 1, C) tensor.
    """
    B, N, C = x.shape

    if last_only:
        t = torch.arange(1, N+1, dtype=torch.float32, device=x.device)  # (N,)

        # Compute weights ONLY for the last time step (t=N -> predict t=N+1)
        weights = torch.exp(-0.5 * ((N - t) / bandwidth) ** 2)  # (N,)
        weights = weights / weights.sum()  # Normalize

        # Weighted sum (efficient einsum)
        return torch.einsum('bnc,n->bc', x, weights).unsqueeze(1)  # (B, 1, C)
    else:
        # Time positions (1, 2, ..., N)
        t = torch.arange(1, N+1, dtype=torch.float32, device=x.device)  # (N,)

        # Compute weights (Gaussian kernel)
        dist = t[-1] - t.unsqueeze(0)                           # Distance to last time step (N, N)
        weights = torch.exp(-0.5 * (dist / bandwidth) ** 2)     # (N, N)
        weights = weights / weights.sum(dim=1, keepdim=True)    # Normalize

        # Weighted sum (parallel over B and C)
        pred = torch.einsum('bnc,nm->bmc', x, weights)          # (B, N, C) @ (N, N) -> (B, N, C)
        return pred[:, -1:].unsqueeze(1)                        # Last step as prediction (B, 1, C)


def moving_average(x, window=3, mode='uniform', last_only=False):
    """Non-parametric moving average.
    Args:
        x: Input tensor (B, N, C).
        window: Number of past steps to average.
        mode: 'uniform' or 'linear' weights.
    Returns:
        Predicted (B, 1, C) tensor.
    """
    B, N, C = x.shape

    if last_only:
        # Select the last 'window' elements along N
        x_window = x[:, -window:, :]  # (B, window, C)

        # Compute weights
        if mode == 'uniform':
            pred = x_window.mean(dim=1)  # (B, C)
        elif mode == 'linear':
            weights = torch.linspace(1, 0, window, device=x.device)  # (window,)
            weights = weights / weights.sum()
            pred = torch.einsum('bwc,w->bc', x_window, weights)  # (B, C)
        else:
            raise ValueError(f"Invalid mode. Choose 'uniform' or 'linear'. Got: {mode}")

        return pred.unsqueeze(1)  # (B, 1, C)
    else:
        # Weight initialization
        if mode == 'uniform':
            weights = torch.ones(window, device=x.device) / window
        elif mode == 'linear':
            weights = torch.linspace(1, 0, window, device=x.device)  # Linear decay
            weights = weights / weights.sum()
        else:
            raise ValueError(f"Invalid mode. Choose 'uniform' or 'linear'. Got: {mode}")

        # Sliding window via convolution
        x_pad = torch.nn.functional.pad(x, pad=(0, 0, window-1, 0))     # (B, N+window-1, C)
        weights = weights.view(1, 1, window, 1).expand(B, C, window, 1) # (B, C, window, 1)
        x_unfold = x_pad.unfold(1, window, step=1)                      # (B, N, C, window)
        pred = (x_unfold * weights).sum(dim=-1)                         # (B, N, C)
        return pred[:, -1:].unsqueeze(1)                                # (B, 1, C)


def quantile_regression(x, quantile=0.5):
    """Non-parametric quantile regression.
    Args:
        x: Input tensor (B, N, C).
        quantile: Quantile value (0.5 = median).
    Returns:
        Predicted (B, 1, C) tensor.
    """
    return torch.quantile(x, quantile, dim=1, keepdim=True)     # (B, 1, C)


class KalmanFilter:
    """
    A PyTorch implementation of the Kalman Filter for 1D time series prediction.
    Supports both scalar (position-only) and vector (position + velocity) state models.

    Args:
        initial_state (float or torch.Tensor): Initial state estimate.
        process_noise (float or torch.Tensor): Process noise variance (scalar) or covariance matrix (vector).
        measurement_noise (float or torch.Tensor): Measurement noise variance (scalar) or covariance matrix (vector).
        use_velocity (bool): If True, uses a state vector [position, velocity]. Default: False.
        device (str): Device to run on ('cpu' or 'cuda'). Default: 'cpu'.
    """

    def __init__(self, initial_state, process_noise=0.01, measurement_noise=0.25, use_velocity=True, device='cuda'):
        self.device = device
        self.use_velocity = use_velocity

        # Initialize state and covariance
        if not use_velocity:
            # Scalar state (random walk)
            self.state = torch.tensor([initial_state], dtype=torch.float32, device=device)  # shape: (1,)
            self.error_cov = torch.eye(1, device=device)  # shape: (1, 1)
            self.A = torch.eye(1, device=device)  # State transition matrix
            self.H = torch.eye(1, device=device)  # Observation matrix
            self.Q = torch.tensor([[process_noise]], device=device)  # Process noise
            self.R = torch.tensor([[measurement_noise]], device=device)  # Measurement noise
        else:
            # Vector state (position + velocity)
            self.state = torch.tensor([[initial_state], [0.0]], dtype=torch.float32, device=device)  # shape: (2, 1)
            self.error_cov = torch.eye(2, device=device)  # shape: (2, 2)
            self.A = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=device)  # State transition matrix
            self.H = torch.tensor([[1.0, 0.0]], device=device)  # Observation matrix
            self.Q = torch.diag(torch.tensor([process_noise, process_noise], device=device))  # Process noise
            self.R = torch.tensor([[measurement_noise]], device=device)  # Measurement noise

    def update(self, measurement):
        """
        Update the Kalman Filter state with a new measurement.

        Args:
            measurement (float or torch.Tensor): Observed value at current timestep.
        """
        measurement = torch.tensor(measurement, device=self.device).float()

        if not self.use_velocity:
            # --- Scalar state update ---
            # 1. Predict
            state_pred = self.A * self.state
            error_cov_pred = self.A @ self.error_cov @ self.A.T + self.Q

            # 2. Update
            kalman_gain = error_cov_pred @ self.H.T @ torch.inverse(self.H @ error_cov_pred @ self.H.T + self.R)
            self.state = state_pred + kalman_gain * (measurement - self.H @ state_pred)
            self.error_cov = (torch.eye(1, device=self.device) - kalman_gain @ self.H) @ error_cov_pred
        else:
            # --- Vector state update ---
            # 1. Predict
            state_pred = self.A @ self.state
            error_cov_pred = self.A @ self.error_cov @ self.A.T + self.Q

            # 2. Update
            kalman_gain = error_cov_pred @ self.H.T @ torch.inverse(self.H @ error_cov_pred @ self.H.T + self.R)
            self.state = state_pred + kalman_gain @ (measurement - self.H @ state_pred)
            self.error_cov = (torch.eye(2, device=self.device) - kalman_gain @ self.H) @ error_cov_pred

    def predict(self):
        """
        Predict the next state without a measurement (a priori estimate).

        Returns:
            torch.Tensor: Predicted value at next timestep (scalar or position component).
        """
        if not self.use_velocity:
            return (self.A @ self.state).item()
        else:
            return (self.A @ self.state)[0].item()
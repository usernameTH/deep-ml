# --- cell 1 ---
import torch
import torch.nn as nn


class RegularizedMLP(nn.Module):
    """MLP with BatchNorm1d and Dropout for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_p: float = 0.3):
        super().__init__()
        # TODO: Build at least two hidden blocks:
        #   Linear -> BatchNorm1d -> ReLU (or similar) -> Dropout
        # Final layer: Linear to 1 logit.
        # Store layers on self (Sequential is fine).
        self.Linear = nn.Linear(input_dim, hidden_dim)
        self.Batch = nn.BatchNorm1d(hidden_dim)
        self.Relu = nn.ReLU()
        self.Dropout = nn.Dropout(dropout_p)
        
        self.Linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.Batch2 = nn.BatchNorm1d(hidden_dim)
        self.Relu2 = nn.ReLU()
        self.Dropout2 = nn.Dropout(dropout_p)
        
        self.Linear3 = nn.Linear(hidden_dim, 1)
        
        self.net = nn.Sequential(self.Linear, self.Batch, self.Relu, self.Dropout, self.Linear2, self.Batch2, self.Relu2, self.Dropout2, self.Linear3)
        
    
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return shape (N,) logits for batch x of shape (N, input_dim)."""
        return self.net(x).squeeze(-1)
        


def train_model(model, X_train, y_train, epochs=150, lr=1e-2):
    """Train model in-place with BCEWithLogitsLoss + Adam. Return model."""
    # TODO:
    # - criterion = nn.BCEWithLogitsLoss()
    # - optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # - loop epochs: zero_grad -> forward -> loss -> backward -> step
    # - y_train is float 0/1 with shape (N,)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    for _ in range(epochs):
        optim.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optim.step()
    return model
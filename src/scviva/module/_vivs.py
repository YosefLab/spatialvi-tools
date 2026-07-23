from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.distributions import Bernoulli, Normal


class ImportanceScoreNet(nn.Module):
    """Predicts ``Y`` from log-CPM-normalized ``X``.

    Ported from ``ImportanceScorer``/``ImportanceScorerLinear`` in VIVS's original JAX
    implementation (Boyeau et al. 2024, :cite:p:`Boyeau24`). The per-cell, per-response
    negative log-likelihood (``all_loss``) is the conditional-randomization-test statistic.
    """

    def __init__(
        self,
        n_input: int,
        n_responses: int,
        n_hidden: int = 128,
        dropout_rate: float = 0.0,
        loss_type: Literal["mse", "binary"] = "mse",
        linear: bool = False,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.linear = linear
        # momentum=0.01 here matches flax's momentum=0.99 (flax's `momentum` is the
        # weight kept on the *old* running stat; torch's is the weight given to the *new* batch).
        if linear:
            self.norm1 = nn.BatchNorm1d(n_input, momentum=0.01, eps=1e-3)
            self.dropout1 = nn.Dropout(0.0)
            self.dense_out = nn.Linear(n_input, n_responses)
        else:
            self.dense1 = nn.Linear(n_input, n_hidden)
            self.norm1 = nn.BatchNorm1d(n_hidden, momentum=0.01, eps=1e-3)
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dense_out = nn.Linear(n_hidden, n_responses)
        # Fixed (non-learned) log-std for the "mse" (Normal) likelihood, matching the
        # original's `self.log_std = 0.0` plain attribute (not a trained flax param).
        self.register_buffer("log_std", torch.zeros(()))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.linear:
            h = self.norm1(x)
            h = self.dropout1(h)
            h = self.dense_out(h)
        else:
            h = self.dense1(x)
            h = self.norm1(h)
            h = torch.nn.functional.leaky_relu(h)
            h = self.dropout1(h)
            h = self.dense_out(h)

        if self.loss_type == "mse":
            all_loss = -Normal(h, torch.exp(self.log_std)).log_prob(y)
        else:
            all_loss = -Bernoulli(logits=h).log_prob(y)
        return {"h": h, "loss": all_loss.mean(), "all_loss": all_loss}

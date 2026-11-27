"""Fuse the Hiking observation history and camera latent with self-attention."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn

from .hiking_observations import (
    HIKING_PROPRIO_FRAME,
    HIKING_PROPRIO_HISTORY,
    unpack_hiking_proprio_history,
)


class PerceptiveAttentionPolicy(nn.Module):
    """One attention block with a readout conditioned on the latest observation."""

    PROPRIO_DIM = HIKING_PROPRIO_HISTORY * HIKING_PROPRIO_FRAME
    LATENT_DIM = 128
    EMBED_DIM = 128
    NUM_HEADS = 4
    FFN_DIM = 256
    NUM_TOKENS = HIKING_PROPRIO_HISTORY + 3
    COMMAND_SLICE = slice(6, 9)

    def __init__(self, action_dim: int = 29):
        super().__init__()
        if action_dim < 1:
            raise ValueError("Action width must be positive")
        self.obs_dim = self.PROPRIO_DIM + self.LATENT_DIM
        self.action_dim = action_dim
        self.precision = "float32"
        self.head_dim = self.EMBED_DIM // self.NUM_HEADS
        self.attention_scale = self.head_dim ** -0.5

        self.proprio_projection = nn.Linear(HIKING_PROPRIO_FRAME, self.EMBED_DIM)
        self.visual_projection = nn.Linear(self.LATENT_DIM, self.EMBED_DIM)
        self.command_projection = nn.Linear(3, self.EMBED_DIM)
        self.position = nn.Parameter(torch.empty(self.NUM_TOKENS, self.EMBED_DIM))
        nn.init.normal_(self.position, std=0.02)

        self.attention_norm = nn.LayerNorm(self.EMBED_DIM)
        self.query = nn.Linear(self.EMBED_DIM, self.EMBED_DIM)
        self.key_value = nn.Linear(self.EMBED_DIM, 2 * self.EMBED_DIM)
        self.attention_output = nn.Linear(self.EMBED_DIM, self.EMBED_DIM)
        self.feedforward_norm = nn.LayerNorm(self.EMBED_DIM)
        self.feedforward = nn.Sequential(
            nn.Linear(self.EMBED_DIM, self.FFN_DIM),
            nn.GELU(),
            nn.Linear(self.FFN_DIM, self.EMBED_DIM),
        )
        self.output = nn.Linear(self.EMBED_DIM, action_dim)
        # The residual readout reaches the action head directly. Start with
        # small joint deviations while the world model fits early experience.
        with torch.no_grad():
            self.output.weight.mul_(0.05)
            self.output.bias.zero_()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing() and (observation.ndim < 1 or observation.shape[-1] != self.obs_dim):
            raise ValueError(f"Expected observations with final width {self.obs_dim}")
        flat = observation.reshape(-1, self.obs_dim)
        proprio = unpack_hiking_proprio_history(flat[:, :self.PROPRIO_DIM])
        latent = flat[:, self.PROPRIO_DIM:]
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.precision == "bf16" and observation.device.type == "cuda" else nullcontext())
        with autocast:
            history_tokens = self.proprio_projection(proprio)
            visual_token = self.visual_projection(latent).unsqueeze(1)
            command_token = self.command_projection(proprio[:, -1:, self.COMMAND_SLICE])
            readout_token = history_tokens[:, -1:] + command_token
            tokens = torch.cat((history_tokens, visual_token, command_token, readout_token), dim=1)
            tokens = tokens + self.position.to(tokens.dtype)
            readout = tokens[:, -1:]

            # The other query outputs of a single self-attention block would
            # be discarded by the action head. Keep all keys and values so
            # the state-conditioned readout can still attend to every token.
            normalized = self.attention_norm(tokens)
            query = self.query(normalized[:, -1:]).reshape(-1, 1, self.NUM_HEADS, self.head_dim).transpose(1, 2)
            key, value = self.key_value(normalized).reshape(
                -1, self.NUM_TOKENS, 2, self.NUM_HEADS, self.head_dim,
            ).permute(2, 0, 3, 1, 4).unbind(0)
            logits = (query @ key.transpose(-1, -2)) * self.attention_scale
            weights = torch.softmax(logits, dim=-1).to(value.dtype)
            attended = (weights @ value).reshape(-1, 1, self.EMBED_DIM)
            fused = readout + self.attention_output(attended)
            fused = fused + self.feedforward(self.feedforward_norm(fused))
            action = self.output(fused)
        if action.dtype in (torch.float16, torch.bfloat16):
            action = action.float()
        return action.reshape(*observation.shape[:-1], self.action_dim)

# gdn.py — Optimized GatedDeltaNet (MLX)

import math

import mlx.core as mx
import mlx.nn as nn


def solve_lower_triangular_neumann(L, B, n_iters=None):
    """Solve (I + L) @ X = B where L is strictly lower-triangular.

    Uses Neumann series with repeated squaring:
        (I + L)^{-1} = sum_{k=0}^{N-1} (-L)^k

    Computed in ceil(log2(N)) matmul steps — all GPU-compatible ops.

    Args:
        L: (..., N, N) strictly lower-triangular matrix
        B: (..., N, D) right-hand side
        n_iters: number of doubling steps (default: ceil(log2(N)))

    Returns:
        X: (..., N, D) solution
    """
    N = L.shape[-1]
    if n_iters is None:
        n_iters = max(1, math.ceil(math.log2(N)))

    # Neumann series via repeated squaring:
    #   S accumulates (I + L)^{-1}
    #   P tracks the current power of (-L)
    neg_L = -L
    P = neg_L  # (-L)^1
    S = mx.eye(N, dtype=L.dtype) + P  # I + (-L) = I - L

    for _ in range(n_iters - 1):
        P = P @ P  # (-L)^{2^k}
        S = S @ (mx.eye(N, dtype=L.dtype) + P)

    return S @ B


class GatedDeltaNet(nn.Module):
    def __init__(self, n_embd, n_head, chunk_size=128):
        super().__init__()
        # Combo experiment: pair winning chunk and numeric changes.
        self.chunk_size = 128
        self.n_head = n_head
        self.W_alpha = nn.Linear(n_embd, n_head, bias=False)
        self.W_beta = nn.Linear(n_embd, n_head, bias=False)
        self.log_temp_alpha = mx.zeros((n_head,), dtype=mx.float32)
        self.log_temp_beta = mx.zeros((n_head,), dtype=mx.float32)
        # Alternative solve: single-step Neumann approximation.
        self._neumann_iters = 1

    def __call__(self, q, k, v, x_raw):
        out_dtype = q.dtype
        q = q.astype(mx.float32)
        k = k.astype(mx.float32)
        v = v.astype(mx.float32)
        x_raw = x_raw.astype(mx.float32)

        B, H, T, D = q.shape
        CS = self.chunk_size

        # --- gates (tanh-based sigmoid surrogate) ---
        alpha_logits = self.W_alpha(x_raw)
        beta_logits = self.W_beta(x_raw)
        alpha_temp = mx.exp(self.log_temp_alpha)[None, None, :]
        beta_temp = mx.exp(self.log_temp_beta)[None, None, :]
        alpha_pre = alpha_temp * alpha_logits
        beta_pre = beta_temp * beta_logits
        alpha = mx.clip(0.5 * (alpha_pre / (1.0 + mx.abs(alpha_pre)) + 1.0), 3e-4, 1.0)
        beta = mx.clip(0.5 * (beta_pre / (1.0 + mx.abs(beta_pre)) + 1.0), 3e-4, 1.0)

        # --- pad to multiple of chunk_size ---
        pad_len = (CS - T % CS) % CS
        if pad_len > 0:
            q = mx.pad(q, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
            k = mx.pad(k, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
            v = mx.pad(v, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
            alpha = mx.pad(alpha, [(0, 0), (0, pad_len), (0, 0)], constant_values=1.0)
            beta = mx.pad(beta, [(0, 0), (0, pad_len), (0, 0)])

        T_pad = q.shape[2]
        NC = T_pad // CS

        # --- reshape into chunks ---
        alpha = alpha.transpose(0, 2, 1).reshape(B, H, NC, CS)
        beta = beta.transpose(0, 2, 1).reshape(B, H, NC, CS)
        q = q.reshape(B, H, NC, CS, D)
        k = k.reshape(B, H, NC, CS, D)
        v = v.reshape(B, H, NC, CS, D)

        # --- constants ---
        S = mx.zeros((B, H, D, D), dtype=mx.float32)
        S_prev = S
        M = mx.tril(mx.ones((CS, CS), dtype=mx.float32))
        inv_d = 1.0 / float(D + 4)
        inv_sqrt_d = 1.0 / (float(D) ** 0.5)
        n_iters = 1
        all_outputs = []

        for c in range(NC):
            if c > 0:
                S = 0.5 * S + 0.5 * S_prev
            Q_c = q[:, :, c]  # (B, H, CS, D)
            K_c = k[:, :, c]
            V_c = v[:, :, c]
            # Throughput experiment: lower-precision K/V path.
            K_c_lp = K_c.astype(mx.bfloat16)
            V_c_lp = V_c.astype(mx.bfloat16)
            a_c = alpha[:, :, c]  # (B, H, CS)
            b_c = beta[:, :, c]

            # ---- 1. Time decay (log-space for numerical stability) ----
            log_gamma = mx.cumsum(mx.log(a_c + 1e-8), axis=-1)
            log_gamma = mx.clip(log_gamma, -40.0, 0.0)
            gamma = mx.exp(log_gamma).astype(mx.bfloat16).astype(mx.float32)
            log_ratio = mx.clip(log_gamma[..., :, None] - log_gamma[..., None, :], -40.0, 0.0)
            Gamma = mx.exp(log_ratio) * M[None, None]

            # ---- 2. K_K overlap matrix ----
            K_K = (K_c_lp @ K_c_lp.transpose(0, 1, 3, 2)).astype(mx.float32) * inv_d

            # ---- 3. Undo (U_tilde): corrected erase vectors from K ----
            U_raw = b_c[..., :, None] * K_c_lp.astype(mx.float32)
            L = mx.tril(b_c[..., :, None] * K_K, -1)
            U_tilde = solve_lower_triangular_neumann(L, U_raw, n_iters=self._neumann_iters)

            # ---- 4. Write (W_tilde): corrected write vectors from V ----
            W_raw = b_c[..., :, None] * V_c_lp.astype(mx.float32)
            L_tilde = mx.tril(b_c[..., :, None] * Gamma * K_K, -1)
            W_tilde = solve_lower_triangular_neumann(L_tilde, W_raw, n_iters=self._neumann_iters)

            # ---- 5. Arrow variables ----
            Q_left = Q_c * gamma[..., :, None]

            # ---- 6. Chunk output ----
            S_t = S.transpose(0, 1, 3, 2)
            init_query = (Q_left.astype(mx.bfloat16) @ S_t.astype(mx.bfloat16)).astype(mx.float32)

            attn = (Q_c @ K_c.transpose(0, 1, 3, 2)) * inv_sqrt_d * M[None, None]
            decayed_attn = (Q_left @ K_c.transpose(0, 1, 3, 2)) * inv_sqrt_d * M[None, None]

            intra_writes = (attn * Gamma) @ W_tilde
            intra_erases = decayed_attn @ (
                (U_tilde.astype(mx.bfloat16) @ S_t.astype(mx.bfloat16)).astype(mx.float32)
            )

            all_outputs.append(init_query + intra_writes - intra_erases)

            # ---- 7. State update ----
            gamma_T = gamma[..., -1:]
            U_left = U_tilde * gamma[..., :, None]
            log_k_ratio = mx.clip(log_gamma[..., -1:] - log_gamma, -40.0, 40.0)
            K_right = K_c * mx.exp(log_k_ratio)[..., None]

            net_new = W_tilde - (U_left.astype(mx.bfloat16) @ S_t.astype(mx.bfloat16)).astype(
                mx.float32
            )
            S = S * gamma_T[..., None] + net_new.transpose(0, 1, 3, 2) @ K_right
            S = mx.clip(S, -3.0, 3.0)
            S_prev = S

        O = mx.concatenate(all_outputs, axis=2)[:, :, :T, :] * (1.0 / ((float(D) + 4.0) ** 0.5))
        O = O
        return O.astype(out_dtype)

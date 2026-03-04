import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class STICBlock(nn.Module):
    """
    Bloco espaço-temporal:
    - atenção temporal por nó
    - mistura espacial guiada pela adjacência
    - feed-forward + residual
    """

    def __init__(self, hidden_dim, num_heads=4, ff_multiplier=2, dropout=0.1):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.spatial_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        ff_dim = hidden_dim * ff_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: [B, T, N, D]
        B, T, N, D = x.shape

        temp_in = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        temp_out, _ = self.temporal_attn(temp_in, temp_in, temp_in, need_weights=False)
        temp_out = temp_out.reshape(B, N, T, D).permute(0, 2, 1, 3)

        spatial_out = torch.einsum("nm,btmd->btnd", adj, x)
        spatial_out = self.spatial_proj(spatial_out)

        gate = torch.sigmoid(self.gate_proj(torch.cat([temp_out, spatial_out], dim=-1)))
        fused = gate * temp_out + (1.0 - gate) * spatial_out

        h = self.norm1(x + self.dropout(fused))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h


class STICformer(nn.Module):
    def __init__(
        self,
        adj_mx,
        num_nodes,
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        seq_len=12,
        horizon=12,
        num_layers=2,
        num_heads=4,
        ff_multiplier=2,
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=50,
        patience=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) deve ser divisível por num_heads ({num_heads})"
            )

        if sp.issparse(adj_mx):
            adj_mx = adj_mx.toarray()
        adj_mx = np.asarray(adj_mx, dtype=np.float32)
        adj_mx = torch.tensor(adj_mx, dtype=torch.float32)
        adj_mx = adj_mx / (adj_mx.sum(dim=1, keepdim=True) + 1e-8)

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.epochs = epochs
        self.patience = patience
        self.device = device

        self.register_buffer("adj", adj_mx)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.hist_time_embedding = nn.Embedding(seq_len, hidden_dim)
        self.fut_time_embedding = nn.Embedding(horizon, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                STICBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.decoder_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.dropout = nn.Dropout(dropout)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.loss_fn = nn.MSELoss()

        self.to(device)

    def _build_embeddings(self, batch_size):
        device = self.device
        node_idx = torch.arange(self.num_nodes, device=device)
        node_emb = self.node_embedding(node_idx)  # [N, D]

        hist_idx = torch.arange(self.seq_len, device=device)
        hist_time = self.hist_time_embedding(hist_idx)  # [T, D]
        hist_ste = hist_time[:, None, :] + node_emb[None, :, :]
        hist_ste = hist_ste.unsqueeze(0).expand(batch_size, -1, -1, -1)

        fut_idx = torch.arange(self.horizon, device=device)
        fut_time = self.fut_time_embedding(fut_idx)  # [H, D]
        fut_ste = fut_time[:, None, :] + node_emb[None, :, :]
        fut_ste = fut_ste.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return hist_ste, fut_ste

    def forward(self, x):
        # x: [B, T, N, C]
        x = x.to(self.device).float()
        B, T, N, C = x.shape

        if T != self.seq_len:
            raise ValueError(f"seq_len esperado={self.seq_len}, recebido={T}")
        if N != self.num_nodes:
            raise ValueError(f"num_nodes esperado={self.num_nodes}, recebido={N}")
        if C != self.input_dim:
            raise ValueError(f"input_dim esperado={self.input_dim}, recebido={C}")

        hist_ste, fut_ste = self._build_embeddings(B)

        h = self.input_proj(x)
        h = self.dropout(h + hist_ste)

        for block in self.blocks:
            h = block(h, self.adj)

        # Decoder por atenção cruzada (por nó)
        memory = h.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        query = fut_ste.permute(0, 2, 1, 3).reshape(B * N, self.horizon, self.hidden_dim)

        dec_out, _ = self.decoder_attn(query, memory, memory, need_weights=False)
        dec_out = self.decoder_norm(query + self.dropout(dec_out))
        dec_out = dec_out.reshape(B, N, self.horizon, self.hidden_dim).permute(0, 2, 1, 3)

        y = self.output_head(dec_out)
        return y

    def fit(self, train_loader, val_loader=None):
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = "best_model_sticformer.pth"
        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0.0

            for X, Y in train_loader:
                X = X.to(self.device).float()
                Y = Y.to(self.device).float()

                self.optimizer.zero_grad()
                Y_pred = self.forward(X)
                loss = self.loss_fn(Y_pred, Y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            self.train_losses.append(float(avg_loss))
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_loss:.4f}")

            if val_loader:
                val_loss = self.evaluate(val_loader)
                self.val_losses.append(float(val_loss))
                print(f"   Val Loss: {val_loss:.4f}")
                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.state_dict(), best_model_path)
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f"Early stopping na época {epoch+1}")
                    break

        if val_loader:
            try:
                self.load_state_dict(torch.load(best_model_path, map_location=self.device))
            except FileNotFoundError:
                pass

    def evaluate(self, loader):
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X, Y in loader:
                X = X.to(self.device).float()
                Y = Y.to(self.device).float()
                Y_pred = self.forward(X)
                total_loss += self.loss_fn(Y_pred, Y).item()
        return total_loss / len(loader)

    def predict(self, loader):
        self.eval()
        preds = []
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(self.device).float()
                preds.append(self.forward(X).cpu())
        return torch.cat(preds, dim=0)

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchSTG(nn.Module):
    def __init__(
        self,
        adj_mx,
        num_nodes,
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        seq_len=12,
        horizon=12,
        patch_len=4,
        patch_stride=2,
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

        if patch_len <= 0 or patch_stride <= 0:
            raise ValueError("patch_len e patch_stride devem ser > 0")
        if seq_len < patch_len:
            raise ValueError(
                f"seq_len ({seq_len}) deve ser >= patch_len ({patch_len})"
            )
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
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.epochs = epochs
        self.patience = patience
        self.device = device

        self.num_patches = 1 + (seq_len - patch_len) // patch_stride
        patch_input_dim = patch_len * input_dim

        self.register_buffer("adj", adj_mx)

        self.patch_proj = nn.Linear(patch_input_dim, hidden_dim)
        self.patch_pos_emb = nn.Embedding(self.num_patches, hidden_dim)
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * output_dim),
        )
        self.dropout = nn.Dropout(dropout)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.loss_fn = nn.MSELoss()

        self.to(device)

    def _patchify(self, x):
        # x: [B, T, N, C] -> [B, N, P, patch_len * C]
        B, T, N, C = x.shape
        patches = []
        for start in range(0, T - self.patch_len + 1, self.patch_stride):
            end = start + self.patch_len
            patch = x[:, start:end]  # [B, patch_len, N, C]
            patch = patch.permute(0, 2, 1, 3).reshape(B, N, self.patch_len * C)
            patches.append(patch)
        return torch.stack(patches, dim=2)

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

        patches = self._patchify(x)  # [B, N, P, patch_dim]
        B, N, P, _ = patches.shape

        patch_tokens = self.patch_proj(patches)  # [B, N, P, D]

        pos_idx = torch.arange(P, device=self.device)
        pos_emb = self.patch_pos_emb(pos_idx).view(1, 1, P, self.hidden_dim)

        node_idx = torch.arange(N, device=self.device)
        node_emb = self.node_emb(node_idx).view(1, N, 1, self.hidden_dim)

        patch_tokens = self.dropout(patch_tokens + pos_emb + node_emb)
        patch_tokens = patch_tokens.reshape(B * N, P, self.hidden_dim)
        patch_tokens = self.encoder(patch_tokens)

        node_repr = patch_tokens.mean(dim=1).reshape(B, N, self.hidden_dim)

        # Integração espacial por grafo
        graph_ctx = torch.einsum("nm,bmd->bnd", self.adj, node_repr)
        node_repr = self.fusion_norm(node_repr + self.graph_proj(graph_ctx))

        y = self.output_head(node_repr)  # [B, N, H * C]
        y = y.view(B, N, self.horizon, self.output_dim).permute(0, 2, 1, 3)
        return y

    def fit(self, train_loader, val_loader=None):
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = "best_model_patchstg.pth"

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
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_loss:.4f}")

            if val_loader:
                val_loss = self.evaluate(val_loader)
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

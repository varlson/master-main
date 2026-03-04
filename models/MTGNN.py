import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixProp(nn.Module):
    """
    Propagacao simples multi-hop inspirada no MTGNN.
    Entrada/saida: [B, C, N, T]
    """

    def __init__(self, in_channels, out_channels, gcn_depth=2, alpha=0.05, dropout=0.0):
        super().__init__()
        self.gcn_depth = gcn_depth
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Conv2d((gcn_depth + 1) * in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x, adj):
        # adj: [N, N]
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)

        out = [x]
        h = x
        for _ in range(self.gcn_depth):
            h = self.alpha * x + (1.0 - self.alpha) * torch.einsum("ij,bcjt->bcit", adj, h)
            out.append(h)

        h_cat = torch.cat(out, dim=1)
        h_cat = self.dropout(h_cat)
        return self.mlp(h_cat)


class TemporalGatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1, dropout=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.filter_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            padding=(0, self.padding),
        )
        self.gate_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            padding=(0, self.padding),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, N, T]
        filt = torch.tanh(self.filter_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        h = filt * gate

        # Remove padding extra para manter comprimento temporal
        if h.size(-1) > x.size(-1):
            h = h[..., -x.size(-1):]

        return self.dropout(h)


class MTGNNBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=2,
        dilation=1,
        gcn_depth=2,
        propalpha=0.05,
        dropout=0.0,
    ):
        super().__init__()
        self.temporal = TemporalGatedConv(
            channels, channels, kernel_size=kernel_size, dilation=dilation, dropout=dropout
        )
        self.mixprop = MixProp(
            channels, channels, gcn_depth=gcn_depth, alpha=propalpha, dropout=dropout
        )
        self.residual = nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x, adj):
        residual = self.residual(x)
        h = self.temporal(x)
        h = self.mixprop(h, adj)
        h = self.norm(h)
        return F.relu(h + residual)


class MTGNN(nn.Module):
    def __init__(
        self,
        adj_mx,
        num_nodes,
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        seq_len=12,
        horizon=12,
        num_blocks=3,
        kernel_size=2,
        dilation_base=2,
        gcn_depth=2,
        propalpha=0.05,
        node_dim=16,
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=50,
        patience=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        if sp.issparse(adj_mx):
            adj_mx = adj_mx.toarray()
        adj_mx = np.asarray(adj_mx, dtype=np.float32)

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.epochs = epochs
        self.patience = patience
        self.device = device

        self.register_buffer("static_adj", torch.tensor(adj_mx, dtype=torch.float32))

        # Grafo adaptativo aprendivel
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, node_dim))
        self.node_emb2 = nn.Parameter(torch.randn(node_dim, num_nodes))

        self.input_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = dilation_base ** i
            self.blocks.append(
                MTGNNBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    gcn_depth=gcn_depth,
                    propalpha=propalpha,
                    dropout=dropout,
                )
            )

        self.end_proj1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))
        self.end_proj2 = nn.Conv2d(hidden_dim, horizon * output_dim, kernel_size=(1, 1))
        self.dropout = nn.Dropout(dropout)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )
        self.loss_fn = nn.MSELoss()

        self.to(device)

    def _adaptive_adj(self):
        adp = torch.matmul(self.node_emb1, self.node_emb2)
        adp = F.softmax(F.relu(adp), dim=1)

        static_adj = self.static_adj
        static_adj = static_adj / (static_adj.sum(dim=1, keepdim=True) + 1e-8)

        # Combina grafo estatico e adaptativo para estabilidade
        return 0.5 * static_adj + 0.5 * adp

    def forward(self, x):
        # x: [B, T, N, C] -> [B, C, N, T]
        x = x.permute(0, 3, 2, 1)

        h = self.input_proj(x)
        adj = self._adaptive_adj()

        for block in self.blocks:
            h = block(h, adj)

        # Usa o ultimo passo temporal apos empilhamento de blocos
        h = h[..., -1:]  # [B, C, N, 1]
        h = F.relu(self.end_proj1(h))
        h = self.dropout(h)
        h = self.end_proj2(h)  # [B, horizon * output_dim, N, 1]

        B = h.size(0)
        h = h.squeeze(-1)  # [B, horizon * output_dim, N]
        h = h.view(B, self.horizon, self.output_dim, self.num_nodes)
        h = h.permute(0, 1, 3, 2).contiguous()  # [B, horizon, N, output_dim]
        return h

    def fit(self, train_loader, val_loader=None):
        self.train()
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = "best_model_mtgnn.pth"
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

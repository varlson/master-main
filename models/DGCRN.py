import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGraphConstructor(nn.Module):
    """
    Constrói uma adjacência dinâmica combinando:
    - adjacência estática normalizada
    - grafo adaptativo aprendível (embeddings de nós)
    """

    def __init__(self, static_adj, num_nodes, node_dim=16):
        super().__init__()
        self.num_nodes = num_nodes

        static_adj = torch.tensor(static_adj, dtype=torch.float32)
        static_adj = static_adj / (static_adj.sum(dim=1, keepdim=True) + 1e-8)
        self.register_buffer("static_adj", static_adj)

        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, node_dim))
        self.node_emb2 = nn.Parameter(torch.randn(node_dim, num_nodes))

        # Gate dinâmico dependente do contexto do passo atual.
        self.context_scale = nn.Parameter(torch.tensor(1.0))
        self.context_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_t):
        # x_t: [B, N, C]
        adaptive_adj = torch.matmul(self.node_emb1, self.node_emb2)
        adaptive_adj = F.softmax(F.relu(adaptive_adj), dim=1)

        context = x_t.mean()
        gate = torch.sigmoid(self.context_scale * context + self.context_bias)
        return gate * self.static_adj + (1.0 - gate) * adaptive_adj


class DGCRNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, gcn_depth=2, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gcn_depth = gcn_depth
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        in_dim = (gcn_depth + 1) * (input_dim + hidden_dim)
        self.w_z = nn.Linear(in_dim, hidden_dim)
        self.w_r = nn.Linear(in_dim, hidden_dim)

        cand_in_dim = (gcn_depth + 1) * (input_dim + hidden_dim)
        self.w_h = nn.Linear(cand_in_dim, hidden_dim)

    def _graph_mix(self, xh, adj):
        # xh: [B, N, F]
        outs = [xh]
        h = xh
        for _ in range(self.gcn_depth):
            h = torch.einsum("nm,bmf->bnf", adj, h)
            outs.append(h)
        return torch.cat(outs, dim=-1)

    def forward(self, x_t, h_prev, adj):
        xh = torch.cat([x_t, h_prev], dim=-1)
        g = self._graph_mix(xh, adj)

        z = torch.sigmoid(self.w_z(g))
        r = torch.sigmoid(self.w_r(g))

        cand_xh = torch.cat([x_t, r * h_prev], dim=-1)
        cand_g = self._graph_mix(cand_xh, adj)
        h_tilde = torch.tanh(self.w_h(cand_g))

        h = (1.0 - z) * h_prev + z * h_tilde
        h = self.norm(h)
        h = self.dropout(h)
        return h


class DGCRN(nn.Module):
    def __init__(
        self,
        adj_mx,
        num_nodes,
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        seq_len=12,
        horizon=12,
        node_dim=16,
        gcn_depth=2,
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

        self.graph_constructor = DynamicGraphConstructor(
            static_adj=adj_mx, num_nodes=num_nodes, node_dim=node_dim
        )
        self.encoder_cell = DGCRNCell(
            input_dim=input_dim, hidden_dim=hidden_dim, gcn_depth=gcn_depth, dropout=dropout
        )
        self.decoder_cell = DGCRNCell(
            input_dim=output_dim, hidden_dim=hidden_dim, gcn_depth=gcn_depth, dropout=dropout
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.loss_fn = nn.MSELoss()

        self.to(device)

    def forward(self, X, Y=None, teacher_forcing_ratio=0.0):
        # X: [B, T, N, C]
        B = X.size(0)
        h = torch.zeros(B, self.num_nodes, self.hidden_dim, device=self.device)

        # Encoder com adjacência dinâmica por passo temporal
        for t in range(self.seq_len):
            x_t = X[:, t]
            adj_t = self.graph_constructor(x_t)
            h = self.encoder_cell(x_t, h, adj_t)

        # Decoder auto-regressivo
        outputs = []
        y_t = torch.zeros(B, self.num_nodes, self.output_dim, device=self.device)

        for t in range(self.horizon):
            adj_t = self.graph_constructor(y_t)
            h = self.decoder_cell(y_t, h, adj_t)
            out_t = self.output_proj(h)
            outputs.append(out_t)

            if Y is not None and np.random.rand() < teacher_forcing_ratio:
                y_t = Y[:, t]
            else:
                y_t = out_t

        return torch.stack(outputs, dim=1)

    def fit(self, train_loader, val_loader=None):
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = "best_model_dgcrn.pth"
        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0.0

            for X, Y in train_loader:
                X = X.to(self.device).float()
                Y = Y.to(self.device).float()

                self.optimizer.zero_grad()
                Y_pred = self.forward(X, Y=Y, teacher_forcing_ratio=0.5)
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

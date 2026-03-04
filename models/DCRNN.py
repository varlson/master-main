import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import mlflow

class DCGRUCell(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, adj_mx, k=2, dropout=0.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.dropout = nn.Dropout(dropout)
        
        if sp.issparse(adj_mx):
            adj_mx = adj_mx.toarray()
        
        # Normalizar adjacência - CORRIGIDO (sem keepdims)
        adj_sum = adj_mx.sum(axis=1, keepdims=True)
        adj_mx = adj_mx / (adj_sum + 1e-8)
        
        # Calcula k poderes da adjacência
        supports = [torch.tensor(adj_mx, dtype=torch.float32)]
        for _ in range(1, k):
            supports.append(torch.matmul(supports[-1], supports[0]))
        
        self.register_buffer('supports', torch.stack(supports))

        in_features = k * input_dim + hidden_dim
        
        self.W_z = nn.Linear(in_features, hidden_dim, bias=True)
        self.W_r = nn.Linear(in_features, hidden_dim, bias=True)
        self.W_h = nn.Linear(in_features, hidden_dim, bias=True)
        
        # Layer normalization para estabilidade
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, h):
        B, N, _ = x.size()
        
        # Difusão espacial
        x_diffused = []
        for i in range(self.k):
            x_diff = torch.einsum('nm,bmf->bnf', self.supports[i], x)
            x_diffused.append(x_diff)
        
        x_cat = torch.cat(x_diffused, dim=-1)
        xh = torch.cat([x_cat, h], dim=-1)
        
        # Gates do GRU
        z = torch.sigmoid(self.W_z(xh))
        r = torch.sigmoid(self.W_r(xh))
        
        xh_r = torch.cat([x_cat, r * h], dim=-1)
        h_tilde = torch.tanh(self.W_h(xh_r))
        
        h_new = (1 - z) * h + z * h_tilde
        h_new = self.layer_norm(h_new)
        h_new = self.dropout(h_new)
        
        return h_new


class DCRNN(nn.Module):
    def __init__(
        self,
        adj_mx,
        num_nodes,
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        seq_len=12,
        horizon=12,
        k=2,
        dropout=0.0,
        lr=1e-3,
        weight_decay=0.0,
        epochs=50,
        patience=10,
        device='cpu',
        use_scheduled_sampling=False,
        teacher_forcing_ratio=0.5
    ):
        super().__init__()

        self.seq_len = seq_len
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.hidden_dim = hidden_dim
        self.use_scheduled_sampling = use_scheduled_sampling
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Encoder-Decoder com dropout
        self.encoder = DCGRUCell(num_nodes, input_dim, hidden_dim, adj_mx, k, dropout)
        self.decoder = DCGRUCell(num_nodes, output_dim, hidden_dim, adj_mx, k, dropout)
        self.fc_output = nn.Linear(hidden_dim, output_dim)

        # Otimizador com weight decay
        self.optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        self.loss_fn = nn.MSELoss()
        self.to(device)

    def forward(self, X, Y=None, teacher_forcing=False):
        B = X.size(0)

        # Encoder
        h = torch.zeros(B, self.num_nodes, self.hidden_dim, device=self.device)
        for t in range(self.seq_len):
            h = self.encoder(X[:, t], h)

        # Decoder com scheduled sampling
        outputs = []
        y_t = torch.zeros(B, self.num_nodes, 1, device=self.device)

        for t in range(self.horizon):
            h = self.decoder(y_t, h)
            out = self.fc_output(h)
            outputs.append(out)
            
            if teacher_forcing and Y is not None and np.random.rand() < self.teacher_forcing_ratio:
                y_t = Y[:, t]
            else:
                y_t = out

        return torch.stack(outputs, dim=1)

    def fit(self, train_loader, val_loader=None):
        self.train()
        best_val_loss = float('inf')
        patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            if self.use_scheduled_sampling:
                self.teacher_forcing_ratio = max(0.0, 0.5 - epoch * 0.01)

            for X, Y in train_loader:
                X = X.to(self.device).float()
                Y = Y.to(self.device).float()

                self.optimizer.zero_grad()
                Y_pred = self.forward(X, Y, teacher_forcing=self.use_scheduled_sampling)
                loss = self.loss_fn(Y_pred, Y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            self.train_losses.append(float(avg_loss))
            
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("learning_rate", self.optimizer.param_groups[0]['lr'], step=epoch)
            
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_loss:.4f}")

            if val_loader:
                val_loss = self.evaluate(val_loader)
                self.val_losses.append(float(val_loss))
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                print(f"   Val Loss: {val_loss:.4f}")
                
                self.scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.state_dict(), 'best_model.pth')
                    mlflow.log_metric("best_val_loss", best_val_loss)
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience:
                    print(f"Early stopping na época {epoch+1}")
                    mlflow.log_metric("early_stop_epoch", epoch)
                    break
        
        if val_loader:
            self.load_state_dict(torch.load('best_model.pth'))

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
                Y_pred = self.forward(X)
                preds.append(Y_pred.cpu())
        return torch.cat(preds, dim=0)
    

import numpy as np
import scipy.sparse as sp
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            padding=self.padding, dilation=dilation
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding] if self.padding > 0 else x


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, adj_mx, k=2):
        super().__init__()
        self.k = k
        
        # ✅ CORREÇÃO: Converter matriz esparsa para densa ANTES de usar
        if sp.issparse(adj_mx):
            adj_mx = adj_mx.toarray()
        
        # Matriz de adjacência normalizada
        adj = torch.tensor(adj_mx, dtype=torch.float32)
        adj_sum = adj.sum(dim=1, keepdims=True)
        adj = adj / (adj_sum + 1e-8)
        
        # Pré-computa k potências
        supports = [adj]
        for _ in range(1, k):
            supports.append(torch.matmul(supports[-1], adj))
        
        self.register_buffer('supports', torch.stack(supports))
        
        # Pesos para diferentes ordens
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_channels, out_channels))
            for _ in range(k)
        ])
        
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # x: [B*T, C, N]
        outputs = []
        
        for i in range(self.k):
            support = self.supports[i]
            # Convolução gráfica
            x_support = torch.einsum('bcn,nm->bcm', x, support)
            # Projeção linear
            out = torch.einsum('bcm,co->bom', x_support, self.weights[i])
            outputs.append(out)
        
        # Combina saídas
        combined = torch.stack(outputs, dim=0).sum(dim=0)
        combined = self.norm(combined)
        combined = self.activation(combined)
        
        return combined


class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super().__init__()
        self.conv1 = DilatedCausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = DilatedCausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # GLU (Gated Linear Unit)
        conv1_out = torch.tanh(self.conv1(x))
        conv2_out = torch.sigmoid(self.conv2(x))
        output = conv1_out * conv2_out
        output = self.norm(output)
        output = self.dropout(output)
        return output


class GraphWaveNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adj_mx, dilation=1, k=2):
        super().__init__()
        self.temporal_conv = TemporalConvLayer(in_channels, out_channels, dilation=dilation)
        self.graph_conv = GraphConvLayer(out_channels, out_channels, adj_mx, k=k)
        
    def forward(self, x):
        # x: [B, C, T, N]
        B, C, T, N = x.shape
        
        # Camada temporal
        x_reshaped = x.permute(0, 3, 1, 2).reshape(B*N, C, T)
        temporal_out = self.temporal_conv(x_reshaped)
        
        # Camada gráfica
        temporal_out_reshaped = temporal_out.reshape(B, N, -1, T).permute(0, 2, 3, 1)
        graph_out = self.graph_conv(
            temporal_out_reshaped.permute(0, 2, 1, 3).reshape(B*T, -1, N)
        )
        
        # Retorna ao formato original
        graph_out = graph_out.reshape(B, T, -1, N).permute(0, 2, 1, 3)
        
        return graph_out + x  # Residual connection


class GraphWaveNet(nn.Module):
    def __init__(
        self,
        adj_mx,
        num_nodes,
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        seq_len=12,
        horizon=12,
        num_blocks=4,
        dilation_base=2,
        k=2,
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=50,
        patience=10,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.horizon = horizon
        self.output_dim = output_dim
        self.device = device
        self.epochs = epochs
        self.patience = patience
        
        # Camada de entrada
        self.input_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))
        
        # Blocos WaveNet
        self.blocks = nn.ModuleList()
        receptive_field = 1
        
        for i in range(num_blocks):
            dilation = dilation_base ** i
            receptive_field += dilation
            
            block = GraphWaveNetBlock(
                hidden_dim,
                hidden_dim,
                adj_mx,
                dilation=dilation,
                k=k
            )
            self.blocks.append(block)
        
        self.receptive_field = receptive_field
        
        # Camadas de saída
        self.output_conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.output_conv2 = nn.Conv1d(hidden_dim, horizon, kernel_size=1)
        self.final_projection = nn.Linear(1, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Otimizador
        self.optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        self.loss_fn = nn.MSELoss()
        self.to(device)
    
    def forward(self, x):
        # x: [B, T, N, C] -> [B, C, T, N]
        x = x.permute(0, 3, 1, 2)
        
        # Camada de entrada
        x = self.input_conv(x)
        
        # Blocos WaveNet
        for block in self.blocks:
            x = block(x)
        
        # Processamento final
        B, C, T, N = x.shape
        
        x = x.permute(0, 3, 1, 2).reshape(B*N, C, T)
        
        x = self.output_conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.output_conv2(x)
        x = x.mean(dim=-1)
        
        x = x.reshape(B, N, self.horizon)
        x = x.permute(0, 2, 1).unsqueeze(-1)
        
        if self.output_dim != 1:
            x = self.final_projection(x)
        
        return x
    
    def fit(self, train_loader, val_loader=None):
        self.train()
        best_val_loss = float('inf')
        patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        best_model_path = Path(getattr(self, "best_model_path", "best_model.pth"))
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X, Y in train_loader:
                X, Y = X.to(self.device), Y.to(self.device)
                
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
                    torch.save(self.state_dict(), str(best_model_path))
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    print(f"Early stopping na época {epoch+1}")
                    break
        
        if val_loader:
            self.load_state_dict(torch.load(str(best_model_path), map_location=self.device))
    
    def evaluate(self, loader):
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X, Y in loader:
                X, Y = X.to(self.device), Y.to(self.device)
                Y_pred = self.forward(X)
                total_loss += self.loss_fn(Y_pred, Y).item()
        return total_loss / len(loader)
    
    def predict(self, loader):
        self.eval()
        preds = []
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(self.device)
                Y_pred = self.forward(X)
                preds.append(Y_pred.cpu())
        return torch.cat(preds, dim=0)

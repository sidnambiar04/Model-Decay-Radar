import numpy as np
import torch
import torch.nn as nn

class RNNMember(nn.Module):
    def __init__(self, input_dim=1, dropout_rate=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.lstm3 = nn.LSTM(64, 64, batch_first=True)
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)
        self.drop3 = nn.Dropout(dropout_rate)
        self.relu  = nn.ReLU()
        self.out   = nn.Linear(64, input_dim)

    def forward(self, x):
        o, _ = self.lstm1(x); o = self.drop1(self.relu(o))
        o, _ = self.lstm2(o); o = self.drop2(self.relu(o))
        o, _ = self.lstm3(o); o = self.drop3(self.relu(o))
        return self.out(o[:, -1, :])

class EnsembleRNNUncertainty:
    def __init__(self, n_members=3, dropout_rate=0.3, seed=42):
        torch.manual_seed(seed)
        self.members = [RNNMember(1, dropout_rate) for _ in range(n_members)]

    @staticmethod
    def make_sequences(series, seq_len=10):
        series = np.asarray(series).reshape(-1)
        X, y = [], []
        for i in range(len(series) - seq_len):
            X.append(series[i:i+seq_len])
            y.append(series[i+seq_len])
        return np.array(X).reshape(-1, seq_len, 1), np.array(y).reshape(-1, 1)

    def train(self, series, seq_len=10, epochs=20, batch_size=64, lr=1e-3, verbose=True):
        X, y = self.make_sequences(series, seq_len)
        Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        n = Xt.shape[0]
        for mi, member in enumerate(self.members):
            opt = torch.optim.Adam(member.parameters(), lr=lr)
            crit = nn.MSELoss()
            member.train()
            for epoch in range(epochs):
                perm = torch.randperm(n)
                el = 0.0
                for i in range(0, n, batch_size):
                    idx = perm[i:i+batch_size]
                    opt.zero_grad()
                    loss = crit(member(Xt[idx]), yt[idx])
                    loss.backward(); opt.step()
                    el += loss.item() * Xt[idx].shape[0]
                if verbose and epoch == epochs - 1:
                    print(f"  [RNN {mi+1}] loss: {el/n:.8f}", flush=True)

    def mc_dropout_predict(self, series, seq_len=10, T=10):
        X, _ = self.make_sequences(series, seq_len)
        Xt = torch.tensor(X, dtype=torch.float32)
        m_means, m_vars = [], []
        for member in self.members:
            member.train()
            passes = []
            with torch.no_grad():
                for _ in range(T):
                    passes.append(member(Xt).numpy())
            passes = np.stack(passes)
            m_means.append(passes.mean(0))
            m_vars.append(passes.var(0))
        return np.mean(m_means, 0), np.mean(m_vars, 0)

    @staticmethod
    def uncertainty_score(unc):
        return unc.reshape(-1)

    @staticmethod
    def normalize_uncertainty(
        variance_scores: np.ndarray,
        baseline_variance: float = 0.0,
        clip_max: float = 1.0,
    ) -> float:
        """
        Derives a normalized [0, 1] epistemic uncertainty score from
        Monte Carlo Dropout predictive forward pass variance.
        """
        arr = np.asarray(variance_scores).reshape(-1)
        if len(arr) == 0:
            return 0.0
        mean_var = float(np.mean(arr))
        if baseline_variance > 1e-9:
            norm = mean_var / baseline_variance
        else:
            norm = 1.0 - float(np.exp(-5.0 * mean_var))
        return float(np.clip(norm, 0.0, clip_max))

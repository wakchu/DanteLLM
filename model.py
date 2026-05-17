import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

# --- CONFIGURAZIONE ---
@dataclass
class GPTConfig:
    """Configurazione del modello GPT."""
    vocab_size: int = 65       # Numero di caratteri unici nel dataset (es. Shakespeare)
    block_size: int = 256      # Lunghezza massima della sequenza (finestra di contesto)
    n_layer: int = 6           # Numero di blocchi Transformer impilati
    n_head: int = 6            # Numero di teste nell'attenzione (n_embd deve essere divisibile per n_head)
    n_embd: int = 384          # Dimensione del vettore di embedding (spazio latente)

# --- ATTENZIONE (IL CUORE) ---
class CausalSelfAttention(nn.Module):
    """Meccanismo di Multi-Head Self-Attention con maschera causale."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Proiezione lineare per ottenere Query, Key e Value in un colpo solo
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Proiezione finale per combinare le teste
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.shape # Batch size, Lunghezza sequenza, Canali (n_embd)

        # 1. Calcolo Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 2. Dividiamo in "teste" per l'attenzione parallela
        # Da (B, T, C) a (B, n_head, T, C/n_head)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Attention con maschera causale (is_causal=True)
        # Questo permette a ogni token di guardare solo i precedenti
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # 4. Ricombiniamo le teste
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 5. Proiezione finale di output
        return self.c_proj(y)

# --- ELABORAZIONE (IL CERVELLO) ---
class MLP(nn.Module):
    """
    Multi-Layer Perceptron: elabora ogni posizione del testo in modo indipendente.
    Mentre l'Attenzione permette ai token di comunicare tra loro, l'MLP è il momento 
    in cui ogni token "riflette" individualmente sulle informazioni raccolte.
    
    Segue una strategia "espandi e contrai":
    1. Espande la dimensione (384 -> 1536) per dare più spazio di calcolo.
    2. Applica la non-linearità GELU per imparare relazioni complesse.
    3. Contrae di nuovo alla dimensione originale (1536 -> 384).
    """
    def __init__(self, config):
        super().__init__()
        # Espandiamo la dimensione di 4 volte per dare al modello "spazio di calcolo"
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # Funzione di attivazione non lineare (una versione più fluida della ReLU)
        self.gelu = nn.GELU(approximate='tanh')
        # Proiettiamo indietro alla dimensione originale
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# --- IL BLOCCO TRANSFORMER ---
class Block(nn.Module):
    """Un singolo blocco che combina Attenzione e MLP con connessioni residue."""
    def __init__(self, config):
        super().__init__()
        # Normalizzazione prima dell'attenzione (Pre-norm)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        # Normalizzazione prima dell'MLP
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Connessione residua: sommiamo l'input all'output (x = x + ...)
        # Questo aiuta i gradienti a fluire meglio durante l'addestramento
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# --- IL MODELLO GPT ---
class GPT(nn.Module):
    """L'architettura completa del modello GPT."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Contenitore per i moduli del Transformer
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Word Token Embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # Word Position Embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # 6 Blocchi
            ln_f = nn.LayerNorm(config.n_embd), # Normalizzazione finale
        ))
        
        # Testa finale che mappa i vettori ai 65 possibili caratteri
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # WEIGHT TYING: condividiamo i pesi tra l'input e l'output
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.shape
        # Creiamo gli indici di posizione [0, 1, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # 1. Recuperiamo gli embedding e sommiamoli
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        x = tok_emb + pos_emb               # Somma con broadcasting (B, T, n_embd)

        # 2. Passiamo attraverso i 6 blocchi Transformer
        for block in self.transformer.h:
            x = block(x)

        # 3. Normalizzazione finale e proiezione ai logit (punteggi caratteri)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        # 4. Calcolo della Loss se abbiamo i target (risposte corrette)
        loss = None
        if targets is not None:
            # Srotoliamo i tensori per la funzione cross_entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

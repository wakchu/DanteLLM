import os
import torch
import math
import json
from tqdm import tqdm
from model import GPTConfig, GPT
from generate import generate

# --- 1. CARICAMENTO DATI ---
def load_data(filepath, block_size, batch_size, device):
    """Carica il testo, crea il vocabolario e prepara i batch."""
    # Percorso assoluto rispetto a questo script
    if not os.path.isabs(filepath):
        base_dir = os.path.dirname(__file__)
        filepath = os.path.join(base_dir, filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Impossibile trovare il file: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Creiamo il vocabolario dei caratteri automaticamente dal testo
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    # Convertiamo tutto il testo in numeri
    tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"Dataset caricato: {len(tokens):,} caratteri, vocabolario: {vocab_size}")

    # Funzione per estrarre un batch di dati casuale
    def get_batch(split_tokens):
        # Scegliamo batch_size indici casuali come punti di partenza
        ix = torch.randint(len(split_tokens) - block_size - 1, (batch_size,))
        # x: input
        x = torch.stack([split_tokens[i:i + block_size] for i in ix]).to(device)
        # y: target (x traslato di 1)
        y = torch.stack([split_tokens[i + 1:i + block_size + 1] for i in ix]).to(device)
        return x, y

    # Dividiamo in 90% training e 10% validation
    n = int(0.9 * len(tokens))
    get_train = lambda: get_batch(tokens[:n])
    get_val = lambda: get_batch(tokens[n:])
    
    return get_train, get_val, vocab_size, stoi, itos

# --- 2. SETUP DISPOSITIVO ---
def get_device():
    """
    Seleziona automaticamente la GPU (CUDA su NVIDIA / MPS su Mac) se disponibile
    e realmente compatibile con la versione installata di PyTorch, altrimenti usa la CPU.
    """
    if torch.cuda.is_available():
        try:
            # Verifica che i kernel CUDA di PyTorch supportino l'architettura della GPU
            # (es. RTX 3050 Ti con Compute Capability 8.6 funziona su GPU;
            #  schede datate come MX130 CC 5.0 fanno fallback sicuro su CPU)
            major, minor = torch.cuda.get_device_capability(0)
            if (major, minor) >= (7, 0):
                _ = torch.zeros(1, device="cuda")
                return torch.device("cuda")
            else:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"Nota: GPU {gpu_name} (CC {major}.{minor}) non supportata dai binari PyTorch correnti, uso CPU.")
        except Exception:
            pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# --- 3. GESTIONE VELOCITÀ DI APPRENDIMENTO (Learning Rate) ---
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Implementa il Cosine Decay con Warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

# --- 4. IL TRAINING LOOP ---
def train(data_path, max_steps=5000, batch_size=64, 
          n_layer=4, n_head=4, n_embd=256, block_size=256, dropout=0.2):
    
    device = get_device()
    if device.type == "cuda":
        print(f"Dispositivo in uso: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Dispositivo in uso: {device}")

    # Carichiamo i dati
    get_train_batch, get_val_batch, vocab_size, stoi, itos = load_data(
        data_path, block_size, batch_size, device
    )

    # Inizializziamo il modello
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    )
    model = GPT(config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modello creato: {n_params / 1e6:.2f}M di parametri (block_size={block_size}, dropout={dropout})")

    # Optimizer con weight_decay più alto (0.1) per contrastare l'overfitting
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    max_lr = 1e-3
    min_lr = max_lr * 0.1
    warmup_steps = 100

    loss_log = {"steps": [], "train": [], "val": []}
    val_loss = 0.0
    best_val_loss = float("inf")

    pbar = tqdm(range(max_steps), desc="Addestramento")
    for step in pbar:
        
        # --- Fase di Validazione ---
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(20):
                    x, y = get_val_batch()
                    _, loss = model(x, y)
                    val_losses.append(loss.item())
                val_loss = sum(val_losses) / len(val_losses)
                is_best = val_loss < best_val_loss
                marker = " *BEST*" if is_best else ""
                tqdm.write(f"Step {step:5d} | Val Loss: {val_loss:.4f}{marker}")

                # Salviamo il modello migliore ogni volta che la Val Loss scende
                if is_best and step > 0:
                    best_val_loss = val_loss
                    torch.save({
                        "step": step,
                        "val_loss": best_val_loss,
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "stoi": stoi,
                        "itos": itos,
                    }, "model_best.pt")
                elif is_best:
                    best_val_loss = val_loss
            model.train()

        # --- Aggiornamento Learning Rate ---
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- Passo di Addestramento ---
        x, y = get_train_batch()
        _, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}", val=f"{val_loss:.4f}", lr=f"{lr:.2e}")

        # --- Salvataggio Log ---
        loss_log["steps"].append(step)
        loss_log["train"].append(loss.item())
        if step % 100 == 0:
            loss_log["val"].append(val_loss)

        # --- Generazione Esempio ---
        if step > 0 and step % 500 == 0:
            model.eval()
            context = "Nel mezzo del cammin"
            sample = generate(model, context, stoi, itos, max_new_tokens=100)
            tqdm.write(f"\n--- Esempio al Passo {step} ---\n{sample}\n---\n")
            model.train()

        # --- Salvataggio Checkpoint ---
        if step > 0 and step % 1000 == 0:
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "config": config,
                "stoi": stoi,
                "itos": itos,
            }, f"checkpoint_{step}.pt")

    # --- Salvataggio Finale ---
    torch.save({
        "step": max_steps,
        "model_state_dict": model.state_dict(),
        "config": config,
        "stoi": stoi,
        "itos": itos,
    }, "model_final.pt")

    with open("loss_log.json", "w") as f:
        json.dump(loss_log, f)

    print(f"Addestramento completato! Miglior modello salvato in 'model_best.pt' (Val Loss: {best_val_loss:.4f}) e ultimo step in 'model_final.pt'")
    return model, stoi, itos

if __name__ == "__main__":
    data_file = "data/divina_commedia.txt"
    train(data_file)

import torch
import argparse
from model import GPT

@torch.no_grad()
def generate(model, prompt, stoi, itos, max_new_tokens=100, temperature=1.0, top_k=None):
    """
    Genera nuovo testo a partire da un prompt in modo autoregressivo.
    """
    model.eval()
    device = next(model.parameters()).device
    # Convertiamo il prompt in una sequenza di numeri, saltando caratteri sconosciuti
    tokens = [stoi[c] for c in prompt if c in stoi]
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Ritagliamo il contesto se supera la block_size del modello
        idx_cond = idx[:, -model.config.block_size:]
        
        # Passiamo i dati al modello per ottenere i punteggi (logits)
        logits, _ = model(idx_cond)
        
        # Prendiamo solo l'ultimo passo temporale e applichiamo la temperatura
        logits = logits[:, -1, :] / temperature
        
        # Top-k filtering: tiene solo i K caratteri più probabili
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # Trasformiamo i punteggi in probabilità (Softmax)
        probs = torch.softmax(logits, dim=-1)
        
        # Estraiamo il prossimo carattere in base alle probabilità (sampling)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Appendiamo il nuovo carattere alla sequenza
        idx = torch.cat((idx, idx_next), dim=1)

    # Convertiamo i numeri finali di nuovo in testo
    out_ids = idx[0].tolist()
    return "".join([itos[i] for i in out_ids])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera testo da un checkpoint GPT addestrato")
    parser.add_argument("checkpoint", help="Percorso del file checkpoint (es. model_final.pt)")
    parser.add_argument("--prompt", default="To be or not", help="Testo iniziale")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Numero di caratteri da generare")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperatura (bassa = deterministico, alta = creativo)")
    parser.add_argument("--top_k", type=int, default=40, help="Campiona solo dai top-k caratteri più probabili")
    parser.add_argument("--seed", type=int, default=None, help="Seed per la riproducibilità")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Caricamento del modello dal checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint["config"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Forziamo l'uso della CPU per compatibilità
    device = torch.device("cpu")
    model.to(device)

    output = generate(model, args.prompt, stoi, itos,
                      max_new_tokens=args.max_new_tokens,
                      temperature=args.temperature,
                      top_k=args.top_k)
    print(f"\n--- Testo Generato ---\n{output}\n----------------------")

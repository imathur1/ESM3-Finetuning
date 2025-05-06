import pandas as pd
import torch
import os

from tqdm import tqdm
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig, LogitsConfig

model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("cuda")
seq_df = pd.read_csv("./data/all.csv")
positive_seqs = seq_df[seq_df.label == 1].seq
negative_seqs = seq_df[seq_df.label == 0].seq

chunk_size = 10000
logits_config = LogitsConfig(sequence=True, return_embeddings=True)

for i in tqdm(range(0, len(positive_seqs), chunk_size)):
    pos_seq_emb_map = {}
    chunk_seqs = positive_seqs[i:i + chunk_size]
    
    for seq in chunk_seqs:
        full_seq = f"EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYC{seq}WGQGTLVTVSS"
        protein = ESMProtein(sequence=full_seq)
        protein_tensor = model.encode(protein)
        logits = model.logits(protein_tensor, logits_config)
        pos_seq_emb_map[seq] = logits.embeddings[:,96:106]
    
    torch.save(pos_seq_emb_map, f"./data/positive_embeddings/seq_emb_checkpoint_{i // chunk_size}.pth")

for i in tqdm(range(0, len(negative_seqs), chunk_size)):
    neg_seq_emb_map = {}
    chunk_seqs = negative_seqs[i:i + chunk_size]
    
    for seq in chunk_seqs:
        full_seq = f"EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYC{seq}WGQGTLVTVSS"
        protein = ESMProtein(sequence=full_seq)
        protein_tensor = model.encode(protein)
        logits = model.logits(protein_tensor, logits_config)
        neg_seq_emb_map[seq] = logits.embeddings[:,96:106]
    
    torch.save(neg_seq_emb_map, f"./data/negative_embeddings/seq_emb_checkpoint_{i // chunk_size}.pth")
import pandas as pd

# From https://zenodo.org/records/10831512
df_hb = pd.read_csv("data/raw/unique_cdr3s_sample_1_hb.csv")
df_mb = pd.read_csv("data/raw/unique_cdr3s_sample_2_mb.csv")
df_lb = pd.read_csv("data/raw/unique_cdr3s_sample_3_lb.csv")

# Add labels
df_hb['label'] = 1
df_mb['label'] = 0
df_lb['label'] = 0

# Combine all dataframes
combined_df = pd.concat([df_hb, df_mb, df_lb], ignore_index=True)
combined_df = combined_df[['aa_cdr3_seq', 'label']]

# Rename the column
combined_df = combined_df.rename(columns={'aa_cdr3_seq': 'seq'})

# Save to a new CSV file
combined_df.to_csv('data/all.csv', index=False)
print(f"Created all.csv with {len(combined_df)} rows") 
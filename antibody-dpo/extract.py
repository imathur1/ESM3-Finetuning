import pandas as pd

# file_path = "data/dpo_data.csv"

# # Load the dataset
# df_full = pd.read_csv(file_path)

# # Sort by similarity in descending order
# df_sorted_full = df_full.sort_values(by="similarity", ascending=False)

# # Extract the top 1M rows
# df_top_1m = df_sorted_full.head(int(1e6))

# # Save the top 1M rows to a new CSV file
# output_file = "data/dpo/top_1m.csv"
# df_top_1m.to_csv(output_file, index=False)

file_path = "data/dpo/top_1m.csv"

# Load the dataset
df_full = pd.read_csv(file_path)

# Randomly sample rows to remove
df_removed_val = df_full.sample(n=100000, random_state=42)

df_remaining = df_full.drop(df_removed_val.index)

df_removed_test = df_remaining.sample(n=10000, random_state=42)

# Create a new DataFrame excluding the sampled rows
df_train = df_remaining.drop(df_removed_test.index)

# Save both DataFrames to separate CSV files
df_removed_val.to_csv("data/dpo/dpo_val_data.csv", index=False)
df_removed_test.to_csv("data/dpo/dpo_test_data.csv", index=False)
df_train.to_csv("data/dpo/dpo_train_data.csv", index=False)
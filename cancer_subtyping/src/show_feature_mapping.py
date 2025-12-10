import pandas as pd
import pickle
import numpy as np

# read the original data
combined_df = pd.read_csv(
    "../data/cohort_1_processed_matrix.csv",
    index_col=0,
    low_memory=False,
)

# read the mapping of protein ID to name
prot_id_to_name = pd.read_csv("../data/all_protein_list_mapping.csv",
                              index_col=0).to_dict()['Gene']

# number of metadata columns
META_COL_NUMS = 4

# get the feature columns (protein columns)
feature_columns = combined_df.columns[META_COL_NUMS-1:]  # 因为index_col=0，所以实际索引要减1

print("="*80)
print("SHAP Feature ID and the corresponding columns in the original table")
print("="*80)
print(f"\nThe column structure of the original CSV file:")
print(f"  Column 0: sample_id (as index)")
print(f"  列 1: cancer_type")
print(f"  Column 2: tissue type")
print(f"  Column 3: cancer subtype")
print(f"  Column 4 and beyond: protein data (UniProt ID)")

print(f"\nAfter reading with pandas (index_col=0):")
all_columns = combined_df.columns.tolist()
print(f"  Column 0: {all_columns[0]}")
print(f"  Column 1: {all_columns[1]}")
print(f"  Column 2: {all_columns[2]}")
print(f"  Column 3 and beyond: protein data")

print(f"\nAfter extracting features with iloc[:, {META_COL_NUMS-1}:]:")
print(f"  Total {len(feature_columns)} features (proteins)")

print("\n" + "="*80)
print("The mapping of the first 20 features:")
print("="*80)
print(f"{'SHAP Index':<12} {'UniProt ID':<15} {'Gene Name':<20} {'Original CSV Column Index':<15}")
print("-"*80)

for i, uniprot_id in enumerate(feature_columns[:20]):
    gene_name = prot_id_to_name.get(uniprot_id, "Unknown")
    original_col_idx = i + 4  # the column index in the original CSV
    print(f"{i:<12} {uniprot_id:<15} {gene_name:<20} {original_col_idx:<15}")

print("\n" + "="*80)
print("The mapping of the last 10 features:")
print("="*80)
print(f"{'SHAP Index':<12} {'UniProt ID':<15} {'Gene Name':<20} {'Original CSV Column Index':<15}")
print("-"*80)

for i, uniprot_id in enumerate(feature_columns[-10:]):
    actual_idx = len(feature_columns) - 10 + i
    gene_name = prot_id_to_name.get(uniprot_id, "Unknown")
    original_col_idx = actual_idx + 4
    print(f"{actual_idx:<12} {uniprot_id:<15} {gene_name:<20} {original_col_idx:<15}")

print("\n" + "="*80)
print("Summary:")
print("="*80)
print(f"""
1. The shape of SHAP values: (number of samples, {len(feature_columns)}, number of classes)
   - The index of the second dimension is the Feature Index

2. The mapping of Feature Index:
   - Feature Index 0 → UniProt ID: {feature_columns[0]} → Gene: {prot_id_to_name.get(feature_columns[0], 'Unknown')}
   - Feature Index 1 → UniProt ID: {feature_columns[1]} → Gene: {prot_id_to_name.get(feature_columns[1], 'Unknown')}
   - Feature Index i → The column {META_COL_NUMS} + i in the original CSV file

3. How to use:
   - If SHAP shows that Feature 0 is important, it corresponds to {feature_columns[0]} ({prot_id_to_name.get(feature_columns[0], 'Unknown')})
   - You can view the data in the {META_COL_NUMS} column in the original CSV file
""")

# save the mapping to a file
mapping_df = pd.DataFrame({
    'SHAP_Feature_Index': range(len(feature_columns)),
    'UniProt_ID': feature_columns,
    'Gene_Name': [prot_id_to_name.get(uid, "Unknown") for uid in feature_columns],
    'Original_CSV_Column_Index': range(4, 4 + len(feature_columns))
})

output_file = "../results/shap_feature_mapping.csv"
mapping_df.to_csv(output_file, index=False)
print(f"\nThe complete mapping has been saved to: {output_file}")
print("="*80)



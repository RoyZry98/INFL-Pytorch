import pandas as pd
import pickle
import numpy as np

# 读取原始数据
combined_df = pd.read_csv(
    "../data/cohort_1_processed_matrix.csv",
    index_col=0,
    low_memory=False,
)

# 读取蛋白质ID到名称的映射
prot_id_to_name = pd.read_csv("../data/all_protein_list_mapping.csv",
                              index_col=0).to_dict()['Gene']

# 元数据列数
META_COL_NUMS = 4

# 获取特征列（蛋白质列）
feature_columns = combined_df.columns[META_COL_NUMS-1:]  # 因为index_col=0，所以实际索引要减1

print("="*80)
print("SHAP Feature ID 与原表格列的对应关系")
print("="*80)
print(f"\n原始CSV文件列结构:")
print(f"  列 0: sample_id (作为index)")
print(f"  列 1: cancer_type")
print(f"  列 2: Tissue type")
print(f"  列 3: Cancer subtype")
print(f"  列 4 及之后: 蛋白质数据 (UniProt ID)")

print(f"\n在 pandas 读取后 (index_col=0):")
all_columns = combined_df.columns.tolist()
print(f"  列 0: {all_columns[0]}")
print(f"  列 1: {all_columns[1]}")
print(f"  列 2: {all_columns[2]}")
print(f"  列 3 及之后: 蛋白质数据")

print(f"\n使用 iloc[:, {META_COL_NUMS-1}:] 提取特征后:")
print(f"  总共 {len(feature_columns)} 个特征（蛋白质）")

print("\n" + "="*80)
print("前20个特征的映射关系:")
print("="*80)
print(f"{'SHAP Index':<12} {'UniProt ID':<15} {'Gene Name':<20} {'原CSV列索引':<15}")
print("-"*80)

for i, uniprot_id in enumerate(feature_columns[:20]):
    gene_name = prot_id_to_name.get(uniprot_id, "Unknown")
    original_col_idx = i + 4  # 在原始CSV中的列索引
    print(f"{i:<12} {uniprot_id:<15} {gene_name:<20} {original_col_idx:<15}")

print("\n" + "="*80)
print("最后10个特征的映射关系:")
print("="*80)
print(f"{'SHAP Index':<12} {'UniProt ID':<15} {'Gene Name':<20} {'原CSV列索引':<15}")
print("-"*80)

for i, uniprot_id in enumerate(feature_columns[-10:]):
    actual_idx = len(feature_columns) - 10 + i
    gene_name = prot_id_to_name.get(uniprot_id, "Unknown")
    original_col_idx = actual_idx + 4
    print(f"{actual_idx:<12} {uniprot_id:<15} {gene_name:<20} {original_col_idx:<15}")

print("\n" + "="*80)
print("总结:")
print("="*80)
print(f"""
1. SHAP values 的形状: (样本数, {len(feature_columns)}, 类别数)
   - 第二维度的索引就是 Feature Index

2. Feature Index 对应关系:
   - Feature Index 0 → UniProt ID: {feature_columns[0]} → Gene: {prot_id_to_name.get(feature_columns[0], 'Unknown')}
   - Feature Index 1 → UniProt ID: {feature_columns[1]} → Gene: {prot_id_to_name.get(feature_columns[1], 'Unknown')}
   - Feature Index i → 原CSV文件的列 {META_COL_NUMS} + i

3. 如何使用:
   - 如果 SHAP 显示 Feature 0 很重要，它对应的是 {feature_columns[0]} ({prot_id_to_name.get(feature_columns[0], 'Unknown')})
   - 可以在原CSV文件中查看第 {META_COL_NUMS} 列的数据
""")

# 保存映射关系到文件
mapping_df = pd.DataFrame({
    'SHAP_Feature_Index': range(len(feature_columns)),
    'UniProt_ID': feature_columns,
    'Gene_Name': [prot_id_to_name.get(uid, "Unknown") for uid in feature_columns],
    'Original_CSV_Column_Index': range(4, 4 + len(feature_columns))
})

output_file = "../results/shap_feature_mapping.csv"
mapping_df.to_csv(output_file, index=False)
print(f"\n完整映射关系已保存到: {output_file}")
print("="*80)



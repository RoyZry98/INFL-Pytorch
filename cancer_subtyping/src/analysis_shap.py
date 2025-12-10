import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np

# 加载数据
data = pd.read_csv("/deltadisk/zhangrongyu/leo/ProCanFDL/results/centralised_pred_cancer_type_INR_False_DP_False_LoRA_False_DR_False.csv")

# 提取类别列表
classes = data.columns[1:-2]  # 从表头提取所有癌症类别列名
n_classes = len(classes)

# 转换真实标签为 one-hot 编码
true_labels = pd.get_dummies(data["true"], columns=classes)

# 准备存储每个类别的 AUROC
per_class_auroc = {}

# 计算 Per-Class AUROC
for class_name in classes:
    # 真实标签 (One-vs-Rest)
    y_true = true_labels[class_name]
    
    # 预测概率
    y_pred = data[class_name]
    
    # 计算 AUROC，处理可能的异常情况（如单一类别）
    try:
        auroc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auroc = np.nan  # 如果无法计算 AUROC，返回 NaN
    per_class_auroc[class_name] = auroc

# 计算 Macro-Averaged AUROC（忽略 NaN）
macro_avg_auroc = np.nanmean(list(per_class_auroc.values()))

# 输出结果
print("Per-Class AUROC:")
for class_name, auroc in per_class_auroc.items():
    print(f"{class_name}: {auroc:.4f}")

print(f"\nMacro-Averaged AUROC: {macro_avg_auroc:.4f}")
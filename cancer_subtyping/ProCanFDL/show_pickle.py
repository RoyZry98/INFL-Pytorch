import pickle
import numpy as np
import pandas as pd
import os
import shap
import matplotlib.pyplot as plt

# 设置文件路径
MODEL_PATH = "/deltadisk/zhangrongyu/leo/ProCanFDL/best_model/INFL_INR_True_DP_False_LoRA_False_DR_False"
TARGET = "cancer_type"
repeat = 9

# 选择要读取的文件
shap_values_file = f"{MODEL_PATH}/{TARGET}_shap_values_rep{repeat}_all.pkl"
shap_obj_file = f"{MODEL_PATH}/{TARGET}_shap_values_rep{repeat}_all_obj.pkl"
feature_names_file = f"{MODEL_PATH}/{TARGET}_feature_names_rep{repeat}_all.pkl"
protein_ids_file = f"{MODEL_PATH}/{TARGET}_protein_ids_rep{repeat}_all.pkl"
class_names_file = f"{MODEL_PATH}/{TARGET}_class_names_rep{repeat}_all.pkl"


# 全局变量，将在 main() 中被正确设置
CLASS_NAMES = None
CLASS_ID_TO_NAME = {}
CLASS_NAME_TO_ID = {}

def show_class_names():
    """
    显示所有类别名称及其对应的索引
    """
    print("\n" + "="*60)
    print("类别名称映射表")
    print("="*60)
    for idx, name in CLASS_ID_TO_NAME.items():
        print(f"  {idx:2d}: {name}")
    print("="*60)
    return CLASS_ID_TO_NAME


def get_class_idx(input_str):
    """
    根据输入的类别名称或索引返回类别索引
    
    参数:
    - input_str: 类别名称或索引（字符串）
    
    返回:
    - class_idx: 类别索引（整数）
    """
    # 尝试将输入转换为整数
    try:
        class_idx = int(input_str)
        if class_idx in CLASS_ID_TO_NAME:
            return class_idx
        else:
            print(f"警告: 索引 {class_idx} 超出范围，使用默认值 0")
            return 0
    except ValueError:
        # 如果无法转换为整数，则尝试作为类别名称查找
        if input_str in CLASS_NAME_TO_ID:
            return CLASS_NAME_TO_ID[input_str]
        else:
            print(f"警告: 未找到类别名称 '{input_str}'，使用默认值 0")
            return 0


def load_and_display_pickle(file_path):
    """
    读取pickle文件并显示基本信息
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"读取文件: {file_path}")
    print(f"{'='*60}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # 显示数据类型
    print(f"数据类型: {type(data)}")
    
    # 如果是numpy数组
    if isinstance(data, np.ndarray):
        print(f"数组形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"最小值: {np.min(data)}")
        print(f"最大值: {np.max(data)}")
        print(f"平均值: {np.mean(data)}")
        print(f"\n前5个样本的统计信息:")
        print(f"形状: {data[:5].shape if len(data) > 5 else data.shape}")
        
    # 如果是列表
    elif isinstance(data, list):
        print(f"列表长度: {len(data)}")
        if len(data) > 0:
            print(f"第一个元素类型: {type(data[0])}")
            if isinstance(data[0], np.ndarray):
                print(f"第一个元素形状: {data[0].shape}")
                print(f"所有元素形状: {[item.shape for item in data[:3]]}")
    
    # 如果是SHAP Explanation对象
    elif hasattr(data, 'values') and hasattr(data, 'base_values'):
        print(f"SHAP Explanation 对象")
        print(f"Values 形状: {data.values.shape}")
        if data.base_values is not None:
            if hasattr(data.base_values, 'shape'):
                print(f"Base values 形状: {data.base_values.shape}")
            else:
                print(f"Base values 长度: {len(data.base_values)}")
        else:
            print(f"Base values: None")
        if hasattr(data, 'data') and data.data is not None:
            print(f"Data 形状: {data.data.shape}")
    
    # 其他类型
    else:
        print(f"其他数据结构")
        if hasattr(data, '__dict__'):
            print(f"属性: {list(data.__dict__.keys())}")
    
    return data


def main():
    """
    主函数：读取并显示SHAP值
    """
    global CLASS_NAMES, CLASS_ID_TO_NAME, CLASS_NAME_TO_ID
    
    print("\n开始读取SHAP相关的pickle文件...")
    
    # 首先读取类别名称（最重要！）
    print("\n" + "="*60)
    print("1. 读取 Class Names (类别名称)")
    print("="*60)
    if os.path.exists(class_names_file):
        CLASS_NAMES = load_and_display_pickle(class_names_file)
        print(f"✓ 成功加载类别名称（已按 LabelEncoder 的字母顺序排序）")
        print(f"类别数量: {len(CLASS_NAMES)}")
        print(f"类别列表:")
        for idx, name in enumerate(CLASS_NAMES):
            print(f"  {idx:2d}: {name}")
    
    # 创建映射
    CLASS_ID_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}
    CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    # 读取shap_values
    print("\n" + "="*60)
    print("2. 读取 SHAP Values")
    print("="*60)
    shap_values = load_and_display_pickle(shap_values_file)
    
    # 读取shap_obj
    print("\n" + "="*60)
    print("3. 读取 SHAP Explanation Object")
    print("="*60)
    shap_obj = load_and_display_pickle(shap_obj_file)
    
    # 读取feature names
    print("\n" + "="*60)
    print("4. 读取 Feature Names (Gene Names)")
    print("="*60)
    feature_names = None
    protein_ids = None
    if os.path.exists(feature_names_file):
        feature_names = load_and_display_pickle(feature_names_file)
        print(f"前5个基因名称: {feature_names[:5] if isinstance(feature_names, list) else 'N/A'}")
    else:
        print(f"⚠️  Feature names 文件不存在: {feature_names_file}")
        print("   可视化将使用默认的 'Feature 0', 'Feature 1' 等")
    
    if os.path.exists(protein_ids_file):
        protein_ids = load_and_display_pickle(protein_ids_file)
        print(f"前5个蛋白质ID: {protein_ids[:5] if isinstance(protein_ids, list) else 'N/A'}")
    else:
        print(f"⚠️  Protein IDs 文件不存在: {protein_ids_file}")
    
    return shap_values, shap_obj, feature_names, protein_ids


def visualize_shap_beeswarm(shap_obj, feature_names=None, max_display=5, class_indices=None, save_format='both'):
    """
    为每个类别生成独立的Beeswarm Plot
    
    参数:
    - shap_obj: SHAP Explanation对象
    - feature_names: 特征名称列表（基因名称）
    - max_display: 显示的最大特征数量
    - class_indices: 要可视化的类别索引列表（如果为None，则可视化所有类别）
    - save_format: 保存格式 ('png', 'svg', 或 'both')
    """
    if shap_obj is None:
        print("错误: 需要提供 shap_obj")
        return None
    
    # 数据预处理
    if hasattr(shap_obj.data, 'cpu'):
        feature_data = shap_obj.data.cpu().numpy()
    else:
        feature_data = shap_obj.data
    
    shap_values = shap_obj.values
    n_samples, n_features, n_classes = shap_values.shape
    
    print("\n" + "="*60)
    print("开始生成Beeswarm Plot可视化")
    print("="*60)
    print(f"样本数: {n_samples}, 特征数: {n_features}, 类别数: {n_classes}")
    
    if feature_names is not None:
        print(f"✓ 使用基因名称标签 (共 {len(feature_names)} 个特征)")
    else:
        print("⚠️  未提供基因名称，将使用默认的 'Feature X' 标签")
    
    # 确保输出目录存在
    output_dir = f"{MODEL_PATH}/shap_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定要可视化的类别
    if class_indices is None:
        class_indices = range(n_classes)
    elif isinstance(class_indices, (int, str)):
        class_indices = [class_indices]
    
    # 转换类别名称为索引
    class_indices_converted = []
    for idx in class_indices:
        if isinstance(idx, str):
            class_indices_converted.append(get_class_idx(idx))
        else:
            class_indices_converted.append(idx)
    
    # 为每个类别创建单独的Beeswarm Plot
    for class_idx in class_indices_converted:
        class_name = CLASS_ID_TO_NAME.get(class_idx, f"Class {class_idx}")
        print(f"\n{'='*60}")
        print(f"正在处理类别 {class_idx}: {class_name}")
        print(f"{'='*60}")
        
        # 提取当前类别的SHAP值
        class_shap_values = shap_values[:, :, class_idx]
        
        # 选择top-k个最重要的特征（去掉"other features"显示）
        mean_abs_shap = np.abs(class_shap_values).mean(axis=0)
        top_k_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
        
        print(f"✓ 选择了前 {max_display} 个最重要的特征（不显示其他特征）")
        
        # 只保留top-k特征的数据
        class_shap_values_topk = class_shap_values[:, top_k_indices]
        feature_data_topk = feature_data[:, top_k_indices]
        
        # 获取top-k特征的名称
        if feature_names is not None:
            feature_names_topk = [feature_names[i] for i in top_k_indices]
        else:
            feature_names_topk = [f"Feature {i}" for i in top_k_indices]
        
        # 打印top-k特征
        print(f"Top {max_display} 特征:")
        for i, idx in enumerate(top_k_indices):
            fname = feature_names[idx] if feature_names is not None else f"Feature {idx}"
            print(f"  {i+1:2d}. {fname:20s}: {mean_abs_shap[idx]:.6f}")
        
        # 计算当前类别的SHAP值范围，用于设置合理的横轴限制
        shap_min = np.min(class_shap_values_topk)
        shap_max = np.max(class_shap_values_topk)
        
        # 添加一些边距
        x_margin = (shap_max - shap_min) * 0.1
        xlim_low = shap_min - x_margin
        xlim_high = shap_max + x_margin
        
        print(f"SHAP值范围: [{shap_min:.4f}, {shap_max:.4f}]")
        print(f"设置的横轴范围: [{xlim_low:.4f}, {xlim_high:.4f}]")
        
        # 创建当前类别的Explanation对象（只包含top-k特征）
        class_explanation = shap.Explanation(
            values=class_shap_values_topk,
            base_values=np.zeros(n_samples),
            data=feature_data_topk,
            feature_names=feature_names_topk
        )
        
        # 创建Beeswarm Plot
        try:
            plt.figure(figsize=(10, 8))
            shap.plots.beeswarm(
                class_explanation,
                max_display=max_display,
                show=False
            )
            
            # 设置自定义横轴范围
            plt.xlim(xlim_low, xlim_high)
            
            plt.title(f'SHAP Beeswarm Plot - {class_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # 保存图表
            base_filename = f"{output_dir}/beeswarm_plot_class_{class_name}"
            
            if save_format.lower() in ['png', 'both']:
                output_file_png = f"{base_filename}.png"
                plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
                print(f"✓ 已保存 PNG: {output_file_png}")
            
            if save_format.lower() in ['svg', 'both']:
                output_file_svg = f"{base_filename}.svg"
                plt.savefig(output_file_svg, format='svg', bbox_inches='tight')
                print(f"✓ 已保存 SVG: {output_file_svg}")
            
            plt.close()
        except Exception as e:
            print(f"✗ 生成Beeswarm Plot失败: {e}")
            plt.close()
        
    
    print(f"\n{'='*60}")
    print(f"Beeswarm可视化完成！所有图表已保存到: {output_dir}")
    print(f"{'='*60}")
    return output_dir


if __name__ == "__main__":
    shap_values, shap_obj, feature_names, protein_ids = main()
    
    # 可以在这里添加更多的分析和可视化
    print("\n" + "="*60)
    print("读取完成！")
    print("="*60)
    print("\n可以使用以下变量进行进一步分析:")
    print("- shap_values: SHAP值数组")
    print("- shap_obj: SHAP Explanation对象")
    print("- feature_names: 基因名称列表")
    print("- protein_ids: 蛋白质ID列表")
    
    # 显示类别名称映射
    show_class_names()

    # Beeswarm Plot - 所有类别
    max_display = int(input("显示的最大特征数量 (默认5): ") or "5")
    
    print("\n选择保存格式:")
    print("1. PNG (默认)")
    print("2. SVG (矢量图，适合出版)")
    print("3. 两种格式都保存")
    format_choice = input("请选择 (1/2/3, 默认3): ").strip() or "3"
    
    format_map = {
        '1': 'png',
        '2': 'svg',
        '3': 'both'
    }
    save_format = format_map.get(format_choice, 'both')
    
    visualize_shap_beeswarm(shap_obj, feature_names=feature_names, 
                            max_display=max_display, 
                            class_indices=None,
                            save_format=save_format)


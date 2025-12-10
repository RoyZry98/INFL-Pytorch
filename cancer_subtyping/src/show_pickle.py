import pickle
import numpy as np
import pandas as pd
import os
import shap
import matplotlib.pyplot as plt

# set the file paths
MODEL_PATH = "/deltadisk/zhangrongyu/leo/ProCanFDL/best_model/INFL_INR_True_DP_False_LoRA_False_DR_False"
TARGET = "cancer_type"
repeat = 9

# 选择要读取的文件
shap_values_file = f"{MODEL_PATH}/{TARGET}_shap_values_rep{repeat}_all.pkl"
shap_obj_file = f"{MODEL_PATH}/{TARGET}_shap_values_rep{repeat}_all_obj.pkl"
feature_names_file = f"{MODEL_PATH}/{TARGET}_feature_names_rep{repeat}_all.pkl"
protein_ids_file = f"{MODEL_PATH}/{TARGET}_protein_ids_rep{repeat}_all.pkl"
class_names_file = f"{MODEL_PATH}/{TARGET}_class_names_rep{repeat}_all.pkl"


# global variables, will be correctly set in main()
CLASS_NAMES = None
CLASS_ID_TO_NAME = {}
CLASS_NAME_TO_ID = {}

def show_class_names():
    """
    show all class names and their corresponding indices
    """
    print("\n" + "="*60)
    print("Class Names Mapping Table")
    print("="*60)
    for idx, name in CLASS_ID_TO_NAME.items():
        print(f"  {idx:2d}: {name}")
    print("="*60)
    return CLASS_ID_TO_NAME


def get_class_idx(input_str):
    """
    return the class index for the given class name or index
    
    Parameters:
    - input_str: class name or index (string)
    
    Returns:
    - class_idx: class index (integer)
    """
    # try to convert the input to an integer
    try:
        class_idx = int(input_str)
        if class_idx in CLASS_ID_TO_NAME:
            return class_idx
        else:
            print(f"Warning: index {class_idx} out of range, using default value 0")
            return 0
    except ValueError:
        # if cannot convert to an integer, try to find the class name
        if input_str in CLASS_NAME_TO_ID:
            return CLASS_NAME_TO_ID[input_str]
        else:
            print(f"Warning: class name '{input_str}' not found, using default value 0")
            return 0


def load_and_display_pickle(file_path):
    """
    read the pickle file and display basic information
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Reading file: {file_path}")
    print(f"{'='*60}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # display the data type
    print(f"数据类型: {type(data)}")
    
    # if it is a numpy array
    if isinstance(data, np.ndarray):
        print(f"Array shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Minimum value: {np.min(data)}")
        print(f"Maximum value: {np.max(data)}")
        print(f"Average value: {np.mean(data)}")
        print(f"\nStatistics of the first 5 samples:")
        print(f"形状: {data[:5].shape if len(data) > 5 else data.shape}")
        
    # if it is a list
    elif isinstance(data, list):
        print(f"List length: {len(data)}")
        if len(data) > 0:
            print(f"First element type: {type(data[0])}")
            if isinstance(data[0], np.ndarray):
                print(f"First element shape: {data[0].shape}")
                print(f"All element shapes: {[item.shape for item in data[:3]]}")
    
    # if it is a SHAP Explanation object
    elif hasattr(data, 'values') and hasattr(data, 'base_values'):
        print(f"SHAP Explanation object")
        print(f"Values shape: {data.values.shape}")
        if data.base_values is not None:
            if hasattr(data.base_values, 'shape'):
                print(f"Base values shape: {data.base_values.shape}")
            else:
                print(f"Base values length: {len(data.base_values)}")
        else:
            print(f"Base values: None")
        if hasattr(data, 'data') and data.data is not None:
            print(f"Data shape: {data.data.shape}")
    
    # other types
    else:
        print(f"Other data structures")
        if hasattr(data, '__dict__'):
            print(f"Attributes: {list(data.__dict__.keys())}")
    
    return data


def main():
    """
    main function: read and display SHAP values
    """
    global CLASS_NAMES, CLASS_ID_TO_NAME, CLASS_NAME_TO_ID
    
    print("\nStarting to read SHAP related pickle files...")
    
    # 首先读取类别名称（最重要！）
    print("\n" + "="*60)
    print("1. Read Class Names (class names)")
    print("="*60)
    if os.path.exists(class_names_file):
        CLASS_NAMES = load_and_display_pickle(class_names_file)
        print(f"✓ Successfully loaded class names (sorted alphabetically by LabelEncoder)")
        print(f"Number of classes: {len(CLASS_NAMES)}")
        print(f"Class list:")
        for idx, name in enumerate(CLASS_NAMES):
            print(f"  {idx:2d}: {name}")
    
    # create the mapping
    CLASS_ID_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}
    CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    # read shap_values
    print("\n" + "="*60)
    print("2. Read SHAP Values")
    print("="*60)
    shap_values = load_and_display_pickle(shap_values_file)
    
    # read shap_obj
    print("\n" + "="*60)
    print("3. Read SHAP Explanation Object")
    print("="*60)
    shap_obj = load_and_display_pickle(shap_obj_file)
    
    # read feature names
    print("\n" + "="*60)
    print("4. Read Feature Names (Gene Names)")
    print("="*60)
    feature_names = None
    protein_ids = None
    if os.path.exists(feature_names_file):
        feature_names = load_and_display_pickle(feature_names_file)
        print(f"First 5 gene names: {feature_names[:5] if isinstance(feature_names, list) else 'N/A'}")
    else:
        print(f"⚠️  Feature names file not found: {feature_names_file}")
        print("    Visualization will use default 'Feature 0', 'Feature 1' etc.")
    
    if os.path.exists(protein_ids_file):
        protein_ids = load_and_display_pickle(protein_ids_file)
        print(f"First 5 protein IDs: {protein_ids[:5] if isinstance(protein_ids, list) else 'N/A'}")
    else:
        print(f"⚠️  Protein IDs file not found: {protein_ids_file}")
    
    return shap_values, shap_obj, feature_names, protein_ids


def visualize_shap_beeswarm(shap_obj, feature_names=None, max_display=5, class_indices=None, save_format='both'):
    """
    generate independent Beeswarm Plot for each class
    
    Parameters:
    - shap_obj: SHAP Explanation object
    - feature_names: feature name list (gene names)
    - max_display: maximum number of features to display
    - class_indices: list of class indices to visualize (if None, visualize all classes)
    - save_format: save format ('png', 'svg', or 'both')
    """
    if shap_obj is None:
        print("Error: shap_obj is required")
        return None
    
    # data preprocessing
    if hasattr(shap_obj.data, 'cpu'):
        feature_data = shap_obj.data.cpu().numpy()
    else:
        feature_data = shap_obj.data
    
    shap_values = shap_obj.values
    n_samples, n_features, n_classes = shap_values.shape
    
    print("\n" + "="*60)
    print("Starting to generate Beeswarm Plot visualization")
    print("="*60)
    print(f"Number of samples: {n_samples}, number of features: {n_features}, number of classes: {n_classes}")
    
    if feature_names is not None:
        print(f"✓ Using gene name labels (共 {len(feature_names)} 个特征)")
    else:
        print("⚠️  No gene names provided, using default 'Feature X' labels")
    
    # ensure the output directory exists
    output_dir = f"{MODEL_PATH}/shap_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # determine the classes to visualize
    if class_indices is None:
        class_indices = range(n_classes)
    elif isinstance(class_indices, (int, str)):
        class_indices = [class_indices]
    
    # convert class names to indices
    class_indices_converted = []
    for idx in class_indices:
        if isinstance(idx, str):
            class_indices_converted.append(get_class_idx(idx))
        else:
            class_indices_converted.append(idx)
    
    # create independent Beeswarm Plot for each class
    for class_idx in class_indices_converted:
        class_name = CLASS_ID_TO_NAME.get(class_idx, f"Class {class_idx}")
        print(f"\n{'='*60}")
        print(f"Processing class {class_idx}: {class_name}")
        print(f"{'='*60}")
        
        # extract the SHAP values for the current class
        class_shap_values = shap_values[:, :, class_idx]
        
        # select the top-k most important features (hide "other features" display)
        mean_abs_shap = np.abs(class_shap_values).mean(axis=0)
        top_k_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
        
        print(f"✓ Selected the top {max_display} most important features (hide other features)")
        
        # only keep the data for the top-k features
        class_shap_values_topk = class_shap_values[:, top_k_indices]
        feature_data_topk = feature_data[:, top_k_indices]
        
        # get the names of the top-k features
        if feature_names is not None:
            feature_names_topk = [feature_names[i] for i in top_k_indices]
        else:
            feature_names_topk = [f"Feature {i}" for i in top_k_indices]
        
        # print the top-k features
        print(f"Top {max_display} features:")
        for i, idx in enumerate(top_k_indices):
            fname = feature_names[idx] if feature_names is not None else f"Feature {idx}"
            print(f"  {i+1:2d}. {fname:20s}: {mean_abs_shap[idx]:.6f}")
        
        # calculate the range of SHAP values for the current class, for setting reasonable axis limits
        shap_min = np.min(class_shap_values_topk)
        shap_max = np.max(class_shap_values_topk)
        
        # add some margin
        x_margin = (shap_max - shap_min) * 0.1
        xlim_low = shap_min - x_margin
        xlim_high = shap_max + x_margin
        
        print(f"SHAP value range: [{shap_min:.4f}, {shap_max:.4f}]")
        print(f"Set axis range: [{xlim_low:.4f}, {xlim_high:.4f}]")
        
        # create the Explanation object for the current class (only include the top-k features)
        class_explanation = shap.Explanation(
            values=class_shap_values_topk,
            base_values=np.zeros(n_samples),
            data=feature_data_topk,
            feature_names=feature_names_topk
        )
        
        # create Beeswarm Plot
        try:
            plt.figure(figsize=(10, 8))
            shap.plots.beeswarm(
                class_explanation,
                max_display=max_display,
                show=False
            )
            
            # set custom axis range
            plt.xlim(xlim_low, xlim_high)
            
            plt.title(f'SHAP Beeswarm Plot - {class_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # save the plot
            base_filename = f"{output_dir}/beeswarm_plot_class_{class_name}"
            
            if save_format.lower() in ['png', 'both']:
                output_file_png = f"{base_filename}.png"
                plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
                print(f"✓ Saved PNG: {output_file_png}")
            
            if save_format.lower() in ['svg', 'both']:
                output_file_svg = f"{base_filename}.svg"
                plt.savefig(output_file_svg, format='svg', bbox_inches='tight')
                print(f"✓ Saved SVG: {output_file_svg}")
            
            plt.close()
        except Exception as e:
            print(f"✗ Failed to generate Beeswarm Plot: {e}")
            plt.close()
        
    
    print(f"\n{'='*60}")
    print(f"Beeswarm visualization completed! All plots saved to: {output_dir}")
    print(f"{'='*60}")
    return output_dir


if __name__ == "__main__":
    shap_values, shap_obj, feature_names, protein_ids = main()
    
    # can add more analysis and visualization here
    print("\n" + "="*60)
    print("Reading completed!")
    print("="*60)
    print("\nUse the following variables for further analysis:")
    print("- shap_values: SHAP values array")
    print("- shap_obj: SHAP Explanation object")
    print("- feature_names: gene name list")
    print("- protein_ids: protein ID list")
    
    # show the class name mapping
    show_class_names()

    # Beeswarm Plot - 所有类别
    max_display = int(input("Maximum number of features to display (default 5): ") or "5")
    
    print("\nSelect save format:")
    print("1. PNG (default)")
    print("2. SVG (vector graphics, suitable for publication)")
    print("3. Both formats")
    format_choice = input("Select (1/2/3, default 3): ").strip() or "3"
    
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


import json
import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score
import argparse

def load_processed_data(file_path):
    """Load and process data from a JSONL file"""
    processed_data = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            processed_data.append(data)
    return processed_data

def rank_fun(entry, indicator, k=5, p=5):
    """Function to calculate the aggregated score"""
    def top_k_mean(values, k):
        """Calculate the mean of the top k elements"""
        if values is None or len(values) == 0:
            return np.nan
        values = np.array(values)
        if len(values) <= k:
            return np.mean(values)
        top_k = np.partition(values, -k)[-k:]
        return np.mean(top_k)
        
    def all_mean(values):
        """Calculate the mean of all elements"""
        if values is None or len(values) == 0:
            return np.nan
        return np.mean(values)

    try:
        if indicator == "topk_prob":
            probs = entry.get("prob", [])
            if not probs:
                return np.nan
            length = len(probs)
            # k = int(0.3 * length)
            return -top_k_mean(-np.log(probs), k)
        elif indicator == "topk_entropy":
            entropy = entry.get("entropy", [])
            length = len(entropy)
            # k = int(0.3 * length)
            return -top_k_mean(entropy, k) if entropy else np.nan
        elif indicator == "topk_logtu":
            eu = entry.get("eu_2", [])
            au = entry.get("au_2", [])
            length = min(len(eu), len(au))
            # k = int(0.3 * length)
            if eu and au:
                combined = np.array(eu) * np.array(au)
                return -top_k_mean(combined, k)
            return np.nan
        elif indicator == "avg_entropy":
            entropy = entry.get("entropy", [])
            return -all_mean(entropy) if entropy else np.nan
        elif indicator == "avg_prob":
            probs = entry.get("prob", [])
            if not probs:
                return np.nan
            return -all_mean(-np.log(probs))
        elif indicator == "avg_logtu":
            eu = entry.get("eu_2", [])
            au = entry.get("au_2", [])
            if eu and au:
                combined = np.array(eu) * np.array(au)
                return -all_mean(combined)
            return np.nan
        else:
            print(f"Unrecognized indicator: {indicator}")
            return np.nan
    except Exception as e:
        print(f"Error calculating indicator {indicator}: {str(e)}")
        return np.nan

def calculate_auroc(processed_data, indicator, k):
    """Calculate AUROC (Area Under the ROC Curve)"""
    labels = np.array([entry['llm_judge'] > 0.50 for entry in processed_data])
    # labels = np.array([entry['bleurt'] > 0.50 for entry in processed_data])
    scores = np.array([rank_fun(entry, indicator, k) for entry in processed_data])
    
    # Filter out invalid values
    valid_mask = ~np.isnan(scores)
    valid_labels = labels[valid_mask]
    valid_scores = scores[valid_mask]
    
    if len(valid_labels) == 0:
        print(f"Warning: No valid data for indicator {indicator}")
        return 0.0
    
    # Compute AUROC
    fpr, tpr, _ = roc_curve(valid_labels, valid_scores)
    return auc(fpr, tpr)

def calculate_aupr(processed_data, indicator, k):
    """Calculate AUPR (Area Under the Precision-Recall Curve)"""
    labels = np.array([entry['llm_judge'] > 0.50 for entry in processed_data])
    scores = np.array([rank_fun(entry, indicator, k) for entry in processed_data])

    valid_mask = ~np.isnan(scores)
    valid_labels = labels[valid_mask]
    valid_scores = scores[valid_mask]

    if len(valid_labels) == 0:
        print(f"Warning: No valid data for indicator {indicator}")
        return 0.0

    return average_precision_score(valid_labels, valid_scores)

def calculate_ece_from_confidences(confidences, correctness_list, n_bins=15):
    """A helper to calculate ECE from pre-computed confidences."""
    if len(confidences) == 0:
        return 1.0 # Return worst-case ECE if no data

    confidences = np.array(confidences)
    correctness_list = np.array(correctness_list)

    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_limits[i]) & (confidences <= bin_limits[i+1])
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correctness_list[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece


def calculate_ece(processed_data, n_bins=15):
    """Calculate Expected Calibration Error (ECE) for sequences based on all tokens."""
    confidences = []
    correctness_list = []

    for entry in processed_data:
        probs = entry.get("prob", [])
        if probs:
            confidences.append(np.mean(probs))
            correctness_list.append(entry['llm_judge'] > 0.50)

    if not confidences:
        return 0.0

    return calculate_ece_from_confidences(confidences, correctness_list, n_bins)


def calculate_topk_ece(processed_data, k=15, n_bins=15):
    """Calculate ECE based on the top-k most uncertain tokens."""
    confidences_topk = []
    correctness_list = []

    for entry in processed_data:
        probs = np.array(entry.get("prob", []))
        entropy = np.array(entry.get("entropy", []))

        if len(probs) > k and len(probs) == len(entropy):
            most_uncertain_indices = np.argsort(entropy)[-k:]
            topk_probs = probs[most_uncertain_indices]
            sequence_confidence = np.mean(topk_probs)
            confidences_topk.append(sequence_confidence)
            correctness_list.append(entry['llm_judge'] > 0.50)
        elif len(probs) > 0:
            confidences_topk.append(np.mean(probs))
            correctness_list.append(entry['llm_judge'] > 0.50)

    return calculate_ece_from_confidences(confidences_topk, correctness_list, n_bins)


def analyze_multiple_models(model_paths, k):
    """Analyze the performance of multiple models and print a table"""
    metrics = ['avg_prob','avg_entropy','avg_logtu','topk_prob', 'topk_entropy', 'topk_logtu', 'ECE (Global)', 'ECE (Top-K)', 'Baseline ACC']
    results = {model_name: {} for model_name in model_paths.keys()}
    aupr_results = {model_name: {} for model_name in model_paths.keys()}
    
    # Iterate over each model and calculate metrics
    for model_name, file_path in model_paths.items():
        processed_data = load_processed_data(file_path)
        
        # Calculate AUROC for each metric
        for metric in metrics:
            if metric not in ['ECE (Global)', 'ECE (Top-K)', 'Baseline ACC']:
                auroc = calculate_auroc(processed_data, metric, k)
                results[model_name][metric] = auroc
                aupr = calculate_aupr(processed_data, metric, k)
                aupr_results[model_name][metric] = aupr

        # Calculate ECE
        ece_global = calculate_ece(processed_data)
        results[model_name]['ECE (Global)'] = ece_global

        # Calculate Top-K ECE
        ece_topk = calculate_topk_ece(processed_data)
        results[model_name]['ECE (Top-K)'] = ece_topk
        
        # Calculate Baseline ACC
        baseline_acc = np.mean([entry['llm_judge'] > 0.50 for entry in processed_data])
        results[model_name]['Baseline ACC'] = baseline_acc
    
    # Print the table
    print(f"k: {k}")
    print("AUROC Results:")
    print("{:<15}".format("Metric"), end=" | ")
    for model_name in model_paths.keys():
        print("{:<10}".format(model_name), end=" | ")
    print("\n" + "-" * (15 + 12 * len(model_paths)))
    
    for metric in metrics:
        print("{:<15}".format(metric), end=" | ")
        for model_name in model_paths.keys():
            print("{:<10.4f}".format(results[model_name].get(metric, np.nan)), end=" | ")
        print()
    
    print("=" * (15 + 12 * len(model_paths)))
    print("AUPR Results:")
    print("{:<15}".format("Metric"), end=" | ")
    for model_name in model_paths.keys():
        print("{:<10}".format(model_name), end=" | ")
    print("\n" + "-" * (15 + 12 * len(model_paths)))
    for metric in metrics:
        if metric not in ['ECE (Global)', 'ECE (Top-K)', 'Baseline ACC']:
            print("{:<15}".format(metric), end=" | ")
            for model_name in model_paths.keys():
                print("{:<10.4f}".format(aupr_results[model_name].get(metric, np.nan)), end=" | ")
            print()
    print("=" * (15 + 12 * len(model_paths)))


def get_model_paths_config(config_name, num_samples=800):
    """Get model paths for evaluation. User should configure these paths according to their setup."""
    dataset_name = "tqa"

    if config_name == "default":
        # Example configuration - replace with your actual model paths
        return {
            "base_model": f"./results/{dataset_name}/base_model_one_pass_gene_merge_{num_samples}_llm1.jsonl",
            "sft_model": f"./results/{dataset_name}/sft_model_one_pass_gene_merge_{num_samples}_llm1.jsonl",
        }
    else:
        raise ValueError(f"Unknown config_name: {config_name}. Please configure your model paths in get_model_paths_config().")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=800)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    model_paths_dict_llm = get_model_paths_config(args.config_name, args.num_samples)
    
    analyze_multiple_models(model_paths_dict_llm, args.k)

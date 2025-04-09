import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from skimage.io import imread

def calculate_metrics(true_mask, pred_mask):
    true_mask = true_mask.flatten()
    pred_mask = pred_mask.flatten()

    # Binary confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_mask, pred_mask).ravel()
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Precision
    precision = tp / (tp + fp)
    
    # Recall
    recall = tp / (tp + fn)
    
    # F1-Score
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # AUC-ROC
    auc_roc = roc_auc_score(true_mask, pred_mask)
    
    # Dice Similarity Coefficient (DSC)
    dsc = 2 * tp / (2 * tp + fp + fn)
    
    # Mean Intersection over Union (MIOU)
    iou = tp / (tp + fp + fn)
    miou = np.mean(iou)
    
    # Assuming Modified DSC is same as DSC for simplicity
    mdsc = dsc  # Replace with specific formula if needed
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC-ROC": auc_roc,
        "DSC": dsc,
        "MDSC": mdsc,
        "MIOU": miou,
    }

def load_masks_from_directory(true_mask_dir, pred_mask_dir):
    true_masks = []
    pred_masks = []
    
    true_mask_files = sorted(os.listdir(true_mask_dir))
    pred_mask_files = sorted(os.listdir(pred_mask_dir))
    
    for true_file, pred_file in zip(true_mask_files, pred_mask_files):
        true_mask = imread(os.path.join(true_mask_dir, true_file))
        pred_mask = imread(os.path.join(pred_mask_dir, pred_file))
        
        true_masks.append(true_mask)
        pred_masks.append(pred_mask)
    
    return np.array(true_masks), np.array(pred_masks)

# Example usage:
# Example usage:
true_mask_dir = r"D:\Thesis\datasets\main_DATASET\AUGMENTED\Test\msks"
pred_mask_dir = r"D:\Thesis\Pytorch-UNet-master\Pytorch-UNet-master\results\new 100"

true_masks, pred_masks = load_masks_from_directory(true_mask_dir, pred_mask_dir)

total_accuracy = 0
num_masks = len(true_masks)

# Calculate metrics for each mask and accumulate accuracy
for true_mask, pred_mask in zip(true_masks, pred_masks):
    metrics = calculate_metrics(true_mask, pred_mask)
    total_accuracy += metrics["Accuracy"]

# Compute the average accuracy
average_accuracy = (total_accuracy / num_masks) * 100

print(f"Average Accuracy: {average_accuracy:.2f}%")

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate common segmentation metrics.
    """
    iou = jaccard_score(y_true, y_pred, average='macro')
    dice = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    
    return iou, dice, precision, recall

def compare_models(true_masks, pred_masks_list, model_names):
    """
    Compare the segmentation results of different models.
    """
    metrics = {
        'IoU': [],
        'Dice': [],
        'Precision': [],
        'Recall': []
    }
    
    for pred_masks in pred_masks_list:
        iou_list, dice_list, precision_list, recall_list = [], [], [], []
        
        for y_true, y_pred in zip(true_masks, pred_masks):
            iou, dice, precision, recall = calculate_metrics(y_true.flatten(), y_pred.flatten())
            print( "iou :", iou, " dice", dice , "Percision ", precision , "recall", recall)
            iou_list.append(iou)
            dice_list.append(dice)
            precision_list.append(precision)
            recall_list.append(recall)
        
        metrics['IoU'].append(np.mean(iou_list))
        metrics['Dice'].append(np.mean(dice_list))
        metrics['Precision'].append(np.mean(precision_list))
        metrics['Recall'].append(np.mean(recall_list))
    
    return metrics

def plot_metrics(metrics, model_names):
    """
    Plot the comparison of segmentation metrics.
    """
    categories = list(metrics.keys())
    
    for category in categories:
        values = metrics[category]
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, values)
        plt.title(f'{category} Comparison')
        plt.xlabel('Model')
        plt.ylabel(category)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig(f'{category}_comparison.png')
        plt.show()

def load_image(file_path):
    """
    Load an image from a given file path.
    """
    if os.path.isfile(file_path) and file_path.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        return img
    return None

# Example usage
if __name__ == "__main__":
    true_masks_path = r"D:\Thesis\datasets\main_DATASET\AUGMENTED\msks\00.png"
    pred_masks_model1_path = r"D:\Thesis\Pytorch-UNet-master\Pytorch-UNet-master\results\output100.png"
    pred_masks_model2_path = r"D:\Thesis\Pytorch-UNet-master\Pytorch-UNet-master\results\output60.png"
    
    true_mask = load_image(true_masks_path)
    pred_mask_model1 = load_image(pred_masks_model1_path)
    pred_mask_model2 = load_image(pred_masks_model2_path)
    
    # Ensure all masks are loaded
    if true_mask is None or pred_mask_model1 is None or pred_mask_model2 is None:
        raise FileNotFoundError("One or more image files could not be loaded.")
    
    true_masks = [true_mask]
    pred_masks_model1 = [pred_mask_model1]
    pred_masks_model2 = [pred_mask_model2]
    
    pred_masks_list = [pred_masks_model1, pred_masks_model2]
    model_names = ['Model 1', 'Model 2']
    
    metrics = compare_models(true_masks, pred_masks_list, model_names)
    plot_metrics(metrics, model_names)

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import jaccard_score, roc_auc_score, recall_score, precision_score, f1_score, accuracy_score
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from unet import UNetExtractor_v2, MSEA_unet_v3, MSEA_unet_v2, UNet
# Custom Dataset for Image Segmentation
# Custom Dataset for Image Segmentation
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Only list .jpg files in the images directory
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_filename = self.images[idx]  # Example: image1.jpg
        mask_filename = img_filename.replace('.jpg', '.png')  # Replace .jpg with .png for mask
        
        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)
        image = Image.open(img_path).convert("RGB")  # Load image
        mask = Image.open(mask_path).convert("L")  # Load mask (binary/gray-scale)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# Define transformations (resize and convert to tensor)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to a fixed size
    transforms.ToTensor(),  # Convert to tensor
])



# Dice Similarity Coefficient (DSC)
def dice_coefficient(y_true, y_pred):
    smooth = 1e-6  # To avoid division by zero
    y_true_flat = y_true.view(-1).float()
    y_pred_flat = y_pred.view(-1).float()
    
    intersection = (y_true_flat * y_pred_flat).sum()
    return (2. * intersection + smooth) / (y_true_flat.sum() + y_pred_flat.sum() + smooth)

# Mean Dice Similarity Coefficient (MDSC) - For multiclass segmentation
# Mean Dice Similarity Coefficient (MDSC) - For multiclass segmentation
def mean_dice_coefficient(y_true, y_pred, num_classes):
    dice_per_class = []
    for i in range(num_classes):
        dice = dice_coefficient(y_true == i, y_pred == i)
        dice_per_class.append(dice.item())  # Convert tensor to Python float (.item() moves it to CPU)
    
    return np.mean(dice_per_class)  # Now dice_per_class is a list of Python floats, safe for np.mean()

# Mean Intersection over Union (MIOU)
def mean_iou(y_true, y_pred, num_classes):
    iou_per_class = []
    for i in range(num_classes):
        y_true_np = y_true.cpu().numpy().flatten() == i
        y_pred_np = y_pred.cpu().numpy().flatten() == i
        iou_per_class.append(jaccard_score(y_true_np, y_pred_np))
    return np.mean(iou_per_class)

# Wrapper to calculate all metrics
# Wrapper to calculate all metrics
def compute_metrics(y_true, y_pred, num_classes):
    # Threshold the predicted probabilities for binary classification
    y_pred = (y_pred > 0.5).float()
    
    # Compute each metric
    dice = dice_coefficient(y_true, y_pred).item() * 100
    mdsc = mean_dice_coefficient(y_true, y_pred, num_classes) * 100
    miou = mean_iou(y_true, y_pred, num_classes) * 100
    
    # Flatten to calculate recall, precision, F1, and accuracy
    y_true_np = y_true.cpu().numpy().flatten()  # Ensure tensor is on CPU before converting to numpy
    y_pred_np = y_pred.cpu().numpy().flatten()  # Ensure tensor is on CPU before converting to numpy
    
    recall = recall_score(y_true_np, y_pred_np) * 100
    precision = precision_score(y_true_np, y_pred_np) * 100
    f1 = f1_score(y_true_np, y_pred_np) * 100
    accuracy = accuracy_score(y_true_np, y_pred_np) * 100
    
    # Calculate AUC-ROC (applicable for binary tasks)
    auc_roc = roc_auc_score(y_true_np, y_pred.cpu().numpy().flatten()) * 100
    
    return dice, mdsc, miou, recall, precision, f1, accuracy, auc_roc


# Example evaluation loop with a PyTorch model
def evaluate_model(model, test_loader, device, num_classes=2):
    model.eval()  # Set model to evaluation mode
    dice_scores, mdsc_scores, miou_scores, recalls, precisions, f1_scores, accuracies, auc_rocs = [], [], [], [], [], [], [], []
    
    with torch.no_grad():  # No need to calculate gradients for evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Apply sigmoid for binary or softmax for multiclass segmentation
            if num_classes == 2:
                predictions = torch.sigmoid(predictions)
            else:
                predictions = F.softmax(predictions, dim=1)
            
            # Calculate metrics for the current batch
            for i in range(images.shape[0]):
                dice, mdsc, miou, recall, precision, f1, accuracy, auc_roc = compute_metrics(labels[i], predictions[i], num_classes)
                
                # Store each batch's results
                dice_scores.append(dice)
                mdsc_scores.append(mdsc)
                miou_scores.append(miou)
                recalls.append(recall)
                precisions.append(precision)
                f1_scores.append(f1)
                accuracies.append(accuracy)
                auc_rocs.append(auc_roc)
    
    # Compute the mean of each metric over the entire test dataset
    mean_dice = np.mean(dice_scores)
    mean_mdsc = np.mean(mdsc_scores)
    mean_miou = np.mean(miou_scores)
    mean_recall = np.mean(recalls)
    mean_precision = np.mean(precisions)
    mean_f1 = np.mean(f1_scores)
    mean_accuracy = np.mean(accuracies)
    mean_auc_roc = np.mean(auc_rocs)
    
    # Return the results
    return {
        "Dice": mean_dice,
        "MDSC": mean_mdsc,
        "MIOU": mean_miou,
        "Recall": mean_recall,
        "Precision": mean_precision,
        "F1-score": mean_f1,
        "Accuracy": mean_accuracy,
        "AUC-ROC": mean_auc_roc
    }

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_dir = r'D:\Thesis\datasets\main_DATASET\LABELED\imgs'  # Replace with your image directory
mask_dir = r'D:\Thesis\datasets\main_DATASET\LABELED\masks'   # Replace with your mask directory

test_dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define the metric functions
model = MSEA_unet_v2(n_channels=3, dim=32,  n_classes=1, )


model.to(device=device)
checkpoint = torch.load(r"D:\Thesis\Pytorch-UNet-master\Pytorch-UNet-master\checkpoints\model_MSEA_V2_100.pth", map_location=device)
mask_values = checkpoint.pop('mask_values', [0, 1])
model.load_state_dict(checkpoint)

metrics = evaluate_model(model, test_loader, device, num_classes=1)

# Print the results
print(f"Dice: {metrics['Dice']:.2f}%")
print(f"MDSC: {metrics['MDSC']:.2f}%")
print(f"MIOU: {metrics['MIOU']:.2f}%")
print(f"Recall: {metrics['Recall']:.2f}%")
print(f"Precision: {metrics['Precision']:.2f}%")
print(f"F1-score: {metrics['F1-score']:.2f}%")
print(f"Accuracy: {metrics['Accuracy']:.2f}%")
print(f"AUC-ROC: {metrics['AUC-ROC']:.2f}%")

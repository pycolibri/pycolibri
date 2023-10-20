import torch
from torchmetrics import MeanSquaredError, MeanAbsoluteError, Accuracy, Precision, Recall
from torchmetrics.image import SpectralAngleMapper, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

"""
metrics.py
====================================
The metrics
"""

def psnr(y_true, y_pred):
    """Calculate Peak Signal to Noise Ratio between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        PSNR between y_true and y_pred.
    """
    psnr_metric = PeakSignalNoiseRatio()
    return psnr_metric(y_pred, y_true)

def ssim(y_true, y_pred):
    """Calculate Structural Similarity Index between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        SSIM between y_true and y_pred.
    """
    ssim_metric = StructuralSimilarityIndexMeasure()
    return ssim_metric(y_pred, y_true)

def mse(y_true, y_pred):
    """Calculate Mean Squared Error between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        MSE between y_true and y_pred.
    """
    mse_metric = MeanSquaredError()
    return mse_metric(y_pred, y_true)

def mae(y_true, y_pred):
    """Calculate Mean Absolute Error between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        MAE between y_true and y_pred.
    """
    mae_metric = MeanAbsoluteError()
    return mae_metric(y_pred, y_true)

def accuracy(y_true, y_pred, num_classes=2):
    """Calculate accuracy between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.
        num_classes: Number of classes.

    Returns:
        Accuracy between y_true and y_pred.
    """
    acc_metric = Accuracy(task="binary" if num_classes == 2 else "multiclass",num_classes=num_classes, threshold=0.5 if num_classes == 2 else None)
    return acc_metric(y_pred, y_true)

def precision(y_true, y_pred, num_classes=2):
    """Calculate precision between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.
        num_classes: Number of classes.

    Returns:
        Precision between y_true and y_pred.
    """
    precision_metric = Precision(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
    return precision_metric(y_pred, y_true)

def recall(y_true, y_pred, num_classes=2):
    """Calculate recall between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.
        num_classes: Number of classes.

    Returns:
        Recall between y_true and y_pred.
    """
    recall_metric = Recall(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
    return recall_metric(y_pred, y_true)

def sam(y_true, y_pred, reduce=None):
    """Calculate Spectral Angle Mapper between org and pred.

    Args:
        y_true: Tensor of actual values. [B x H x W x C] 
        y_pred: Tensor of predicted values. [B x H x W x C]

    Returns:
        SAM between org and pred.
    """
    
    reduction = {
        "mean": "elementwise_mean",
        "sum": "sum",
        None: None
    }

    if reduce not in reduction.keys():
        raise ValueError(f"Invalid reduction type. Expected one of: {list(reduction.keys())}")

    sam_metric = SpectralAngleMapper(reduction=reduction[reduce])
    angles = sam_metric(y_pred, y_true)

    return angles
    


if __name__ == "__main__":
    B, C, H, W = 8, 3, 256, 256 


    y_true = torch.rand(B, C, H, W)
    y_pred = torch.rand(B, C, H, W)


    print("PSNR:", psnr(y_true, y_pred).item())
    print("SSIM:", ssim(y_true, y_pred).item())
    print("MSE:", mse(y_true, y_pred).item())
    print("MAE:", mae(y_true, y_pred).item())


    y_true_class = torch.randint(0, 2, (B,)).float()
    y_pred_class = torch.rand(B)

    print("Accuracy:", accuracy(y_true_class, y_pred_class).item())
    print("Precision:", precision(y_true_class, y_pred_class, num_classes=2).item())
    print("Recall:", recall(y_true_class, y_pred_class, num_classes=2).item())
    
    print("SAM (media):", sam(y_true, y_pred, reduce='mean').item())

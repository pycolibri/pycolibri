import torch
from torchmetrics.functional import mean_squared_error, mean_absolute_error, accuracy as acc, precision as prec, recall as rec
from torchmetrics.functional.image import spectral_angle_mapper, peak_signal_noise_ratio, structural_similarity_index_measure

"""
metrics.py
====================================
The metrics
"""

def psnr(y_true, y_pred, data_range=None):
    """Calculate Peak Signal to Noise Ratio between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        PSNR between y_true and y_pred.
    """
    return peak_signal_noise_ratio(y_pred, y_true, data_range=None)

def ssim(y_true, y_pred, data_range=None):
    """Calculate Structural Similarity Index between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        SSIM between y_true and y_pred.
    """
    return structural_similarity_index_measure(y_pred, y_true, data_range=None)

def mse(y_true, y_pred):
    """Calculate Mean Squared Error between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        MSE between y_true and y_pred.
    """
    return mean_squared_error(y_pred, y_true)

def mae(y_true, y_pred):
    """Calculate Mean Absolute Error between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        MAE between y_true and y_pred.
    """
    return mean_absolute_error(y_pred, y_true)

def accuracy(y_true, y_pred, num_classes=2):
    """Calculate accuracy between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.
        num_classes: Number of classes.

    Returns:
        Accuracy between y_true and y_pred.
    """
    return acc(y_pred, y_true, task="binary" if num_classes == 2 else "multiclass",num_classes=num_classes, threshold=0.5 if num_classes == 2 else None)

def precision(y_true, y_pred, num_classes=2):
    """Calculate precision between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.
        num_classes: Number of classes.

    Returns:
        Precision between y_true and y_pred.
    """
    return prec(y_pred, y_true, task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
    

def recall(y_true, y_pred, num_classes=2):
    """Calculate recall between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.
        num_classes: Number of classes.

    Returns:
        Recall between y_true and y_pred.
    """
    return rec(y_pred, y_true, task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)


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

    return spectral_angle_mapper(y_pred, y_true, reduction=reduction[reduce])
    


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


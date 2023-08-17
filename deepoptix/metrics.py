import tensorflow as tf

def psnr(y_true, y_pred, max_val):
  """Calcula el Peak Signal to Noise Ratio entre y_true y y_pred.

  Args:
    y_true: Tensor de valores reales.
    y_pred: Tensor de valores predichos.  
    max_val: Valor máximo de los tensores.

  Returns:
    PSNR entre y_true y y_pred.
  """
  return tf.image.psnr(y_true, y_pred, max_val)

def ssim(y_true, y_pred, max_val):
  """Calcula el Structural Similarity Index entre y_true y y_pred.

  Args:
    y_true: Tensor de valores reales.
    y_pred: Tensor de valores predichos.
    max_val: Valor máximo de los tensores.

  Returns:
    SSIM entre y_true y y_pred.
  """
  return tf.image.ssim(y_true, y_pred, max_val)

def mse(y_true, y_pred):
  """Calcula el Mean Squared Error entre y_true y y_pred.

  Args:
    y_true: Tensor de valores reales.
    y_pred: Tensor de valores predichos.

  Returns:
    MSE entre y_true y y_pred.
  """
  return tf.keras.losses.mean_squared_error(y_true, y_pred) 

def mae(y_true, y_pred):
  """Calcula el Mean Absolute Error entre y_true y y_pred.

  Args:
    y_true: Tensor de valores reales.
    y_pred: Tensor de valores predichos.

  Returns:
    MAE entre y_true y y_pred.
  """
  return tf.keras.losses.mean_absolute_error(y_true, y_pred)

def accuracy(y_true, y_pred, num_classes=2):
  """Calcula la precisión entre y_true y y_pred.

  Args:
    y_true: Tensor de valores reales.
    y_pred: Tensor de valores predichos.
    num_classes: Número de clases.

  Returns:
    Precisión entre y_true y y_pred.
  """
  if num_classes == 2:
    return tf.keras.metrics.binary_accuracy(y_true, y_pred)
  else:
    return tf.keras.metrics.categorical_accuracy(y_true, y_pred)
  

def precision(y_true, y_pred):
  """Calcula la precisión entre y_true y y_pred.

  Args:
    y_true: Tensor de valores reales.
    y_pred: Tensor de valores predichos.

  Returns:
    Precisión entre y_true y y_pred.
  """
  return tf.keras.metrics.precision(y_true, y_pred)

def recall(y_true, y_pred):
  """Calcula la sensibilidad entre y_true y y_pred.

  Args:
    y_true: Tensor de valores reales.
    y_pred: Tensor de valores predichos.

  Returns:
    Sensibilidad entre y_true y y_pred.
  """
  return tf.keras.metrics.recall(y_true, y_pred)
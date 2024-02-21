import itertools
import numpy as np
from sklearn.metrics import confusion_matrix

def create_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15):
  """
  Description: Generates a confusion matrix plot to visualize the performance of a classification model.

  Parameters:
  y_true (array-like): Ground truth (correct) target values.
  y_pred (array-like): Predicted target values returned by the classifier.
  classes (array-like, optional): List of class labels to use for plotting. Defaults to None, in which case integer labels are used.
  figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (10, 10).
  text_size (int, optional): Font size for text annotations in the confusion matrix. Defaults to 15.
  
  Returns: None
  """

  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  n_classes = cm.shape[0]

  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes),
         yticks=np.arange(n_classes),
         xticklabels=labels,
         yticklabels=labels)

  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  threshold = (cm.max() + cm.min()) / 2.

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} \n({cm_norm[i, j]*100:.1f}%)",
             horizontalalignment="center",
             verticalalignment="center",
             color="white" if cm[i, j] > threshold else "black",
             size=text_size)
  plt.savefig('confusion_matrix.png', bbox_inches='tight')

"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""


import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils import class_weight
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    
    # _, labels = torch.max(labels, -1)
    # _, logits = torch.max(logits, -1)
    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits.float(), target = labels.float(),reduction = "none")
    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))
        # pt = torch.exp(-BCLoss) # prevents nans when probability 0
        # F_loss = alpha * (1-pt)**gamma * BCLoss
        # return F_loss.mean()
        # modulator = torch.pow()
    #here
    loss = modulator * BCLoss 
    
    weighted_loss = alpha * loss
    
    focal_loss = torch.sum(weighted_loss)
    
    # focal_loss /= torch.sum(labels)
    # return torch.sum(loss)/torch.sum(labels)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    assert samples_per_cls[0] != 0 or samples_per_cls[1] != 0, "zero sample per classes "
    effective_num = 1.0 - np.power(beta, samples_per_cls)#effective_num has more value for lower sample_per_class
    # print("sample per class ", samples_per_cls, "effective num", effective_num, "power ", np.power(beta, samples_per_cls))
    weights = (1.0 - beta) / np.array(effective_num)
    # print("weights after 1-beta \n", weights)
    #normalized
    tsum = weights[0].item()+weights[1].item()
    # print("tsum", tsum)
    weights = (weights / tsum * no_of_classes)
    # print("*******\n", weights, weights)

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor([weights[0].item(), weights[1].item()]).to( torch.device('cuda' if torch.cuda.is_available() else 'cpu')).float()
    
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot    
    # print(weights, samples_per_cls)
    weights = weights.sum(1)
    # print(weights)
    weights = weights.unsqueeze(1)
    
    weights = weights.repeat(1,no_of_classes)
    
    if loss_type == "focal":
        # print(weights, weights.shape, samples_per_cls )
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        # print(pred.shape, labels_one_hot.shape, weights.shape)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    
    return cb_loss



def fcbl(logits, labels):
    no_of_classes = 2
    beta = 0.9
    gamma = 10
    num_of_ones = torch.sum(labels)
    samples_per_cls = [labels.shape[0]-num_of_ones, num_of_ones]
    
    samples_per_cls = [1,1]
    loss_type = "focal"
    # labels_numpy = labels.cpu().detach().numpy()
    # class_weight = class_weight.compute_class_weight('balanced',[0,1], labels_numpy)
    cb_loss = CB_loss(labels, logits, samples_per_cls, no_of_classes,loss_type, beta, gamma)
    assert not torch.isnan(cb_loss).cpu().numpy(), ["nan loss", samples_per_cls]
    # print("before loss return ", cb_loss)
    return(cb_loss)

def focal_loss(ham_scores, y_true):
  n_classes = 2
  alpha, gamma = 0.25, 2
  # _, y_pred = torch.max(ham_scores, -1)
  y = torch.zeros(ham_scores.shape[0], n_classes).to( torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
  y[range(y.shape[0]), y_true]=1
  BCE_loss = F.binary_cross_entropy(F.sigmoid(ham_scores), y, reduce=False)
  pt = torch.exp(-BCE_loss)
  F_loss = alpha * (1-pt)**gamma * BCE_loss
  
  return torch.sum(F_loss)#torch.mean(F_loss)

import torch
import torch.nn.functional as F
import numpy as np

### One hot the given segmentation mask
def oh(pred):
    pred = pred.float()
    pred = torch.argmax(torch.softmax(pred - torch.max(pred, dim=1, keepdim=True)[0], dim=1), dim=1)
        
    y0 = torch.where(pred==0, 1, 0)
    y1 = torch.where(pred==1, 1, 0)
    y2 = torch.where(pred==2, 1, 0)
    y3 = torch.where(pred==3, 1, 0)
    pred = torch.stack([y0, y1, y2, y3], dim=1).float()
        
    return pred

### Return dice score of each class and average dice score for a given pair of pred/target
def DiceScore(pred, target, smooth=1e-5, mode='batch'):
    class_dice = np.zeros(4)
    #pred and target dimension is (40, 4, 128, 128)
    pred = pred.float()
    target = target.float()
    pred = oh(pred)
    num_slice = pred.shape[0]
    
    for i in range(pred.shape[1]):
        # Dice score over entire batch
        if mode == 'batch':
            a = pred[:,i,:,:].reshape(-1)
            b = target[:,i,:,:].reshape(-1)
            inter = (a * b).sum()
            union = a.sum() + b.sum()
            class_dice[i] = (2. * inter + smooth) / (union + smooth)
            class_dice[i] = class_dice[i].item()

        # Average dice score over all slices
        elif mode == 'sample':
            for j in range(num_slice):
                a = pred[j,i,:,:].reshape(-1)
                b = target[j,i,:,:].reshape(-1)
                inter = (a * b).sum()
                union = a.sum() + b.sum()
                class_dice[i] += ((2. * inter + smooth) / (union + smooth)).item()
            class_dice[i] /= num_slice
    
    avg_dice = np.mean(class_dice)
    
    return avg_dice, class_dice

### Consider implementing mean IoU here
def mean_iou(pred, target, smooth=1e-5):
    pass

########## TESTING ##########
### Dice score but used probability instead of argmax
def DiceScoreTest(pred, target, smooth=1e-5, mode='all'):
    class_dice = np.zeros(4)
    #pred and target dimension is (40, 4, 128, 128)
    pred = pred.float()
    target = target.float()
    pred = torch.softmax(pred - torch.max(pred, dim=1, keepdim=True)[0], dim=1)
    num_slice = pred.shape[0]
    
    for i in range(pred.shape[1]):
        if mode == 'all':
            a = pred[:,i,:,:].reshape(-1)
            b = target[:,i,:,:].reshape(-1)

            inter = (a * b).sum()
            union = a.sum() + b.sum()
            if union == 0:
                class_dice[i] = 1
            else:
                class_dice[i] = (2. * inter + smooth) / (union + smooth)
                class_dice[i] = class_dice[i].item()
        elif mode == 'each':
            for j in range(num_slice):
                a = pred[j,i,:,:].reshape(-1)
                b = target[j,i,:,:].reshape(-1)

                inter = (a * b).sum()
                union = a.sum() + b.sum()
                if union == 0:
                    class_dice[i] = 1
                else:
                    class_dice[i] += ((2. * inter + smooth) / (union + smooth)).item()
            class_dice[i] /= num_slice
    # class_dice[1:] += np.array([0.15, 0.2, 0.2])
    avg_dice = np.mean(class_dice[1:])
    
    return avg_dice, class_dice
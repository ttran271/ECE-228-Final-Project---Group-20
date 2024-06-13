import torch
import torch.nn.functional as F
import numpy as np

#################### READY ####################
########## START ##########
########## Dice Score ##########
def DiceScore(pred, target, smooth=1e-5, mode='all'):
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
    
    avg_dice = np.mean(class_dice[1:])
    
    return avg_dice, class_dice
########## END ##########
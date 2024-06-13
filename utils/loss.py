import torch
import torch.nn as nn
import torch.nn.functional as F

#################### READY ####################
########## START ##########
########## Dice Loss ##########
class DiceLoss(nn.Module):
    def __init__(self, mode='batch'):
        super(DiceLoss, self).__init__()
        self.mode = mode
    
    def forward(self, pred, target, smooth=1e-5):
        total_loss = 0.0

        pred = pred.float()
        pred = torch.softmax(pred - torch.max(pred, dim=1, keepdim=True)[0], dim=1)
        target = target.float()
        num_slice = pred.shape[0]

        for i in range(1, pred.shape[1]):
            if self.mode == 'batch':
                a = pred[:,i,:,:].reshape(-1)
                b = target[:,i,:,:].reshape(-1)
            
                inter = (a * b).sum()
                union = a.sum() + b.sum()
                if union == 0:
                    score = torch.tensor(1, device='cuda:0' if torch.cuda.is_available() else 'cpu')
                else:
                    score = (2. * inter + smooth) / (union + smooth)
                if union == 0:
                    score = torch.tensor(1, device='cuda:0' if torch.cuda.is_available() else 'cpu')
                total_loss += (1-score)

            elif self.mode == 'sample':
                samples_loss = 0.0
                for j in range(num_slice):
                    a = pred[j,i,:,:].reshape(-1)
                    b = target[j,i,:,:].reshape(-1)

                    inter = (a * b).sum()
                    union = a.sum() + b.sum()
                    if (inter == 0) & (union == 0):
                        score = torch.tensor(1, device='cuda:0' if torch.cuda.is_available() else 'cpu')
                    else:
                        score = (2. * inter + smooth) / (union + smooth)
                    samples_loss += (1-score)
                total_loss += (samples_loss / num_slice)

        diceLoss = total_loss
        
        return diceLoss
########## END ##########

########## START ##########
########## Focal Loss ##########
class FocalLoss(nn.Module):
    def __init__(self, alpha=[0.25, 0.8, 0.8, 0.8], gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha, device='cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.reduction = reduction
        self.CE = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')
        
    def un_oh_encode(self, target):
        target = target.float()
        target = torch.argmax(target, dim=1)
        target = target.long()
        
        return target

    def forward(self, pred, target):
        pred = pred.float()
        
        target_index = torch.argmax(target, dim=1).long()
        CE_loss = self.CE(pred, target_index)

        pt = torch.exp(-CE_loss)
        focal_loss = (1-pt)**self.gamma * CE_loss
        
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        
        return focal_loss
########## END ##########

########## START ##########
########## Focal + Dice ##########
class FocalDice(nn.Module):
    def __init__(self, alpha=[0.25, 0.8, 0.8, 0.8], gamma=2.0, reduction='mean', s1=1, s2=1):
        super(FocalDice, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.focal = FocalLoss2(self.alpha, self.gamma, self.reduction)
        self.dice = DiceLoss()
        self.s1 = s1
        self.s2 = s2

    def forward(self, pred, target):
        loss = self.s1*self.focal(pred, target) + self.s2*self.dice(pred, target)
        return loss
########## END ##########
        
#################### TESTING ####################
########## START ##########
########## Old Focal Loss with [alpha, 1-alpha] as weights ##########
class FocalLossTest(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLossTest, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha], device='cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.reduction = reduction
        
    def un_oh_encode(self, target):
        target = target.float()
        target = torch.argmax(target, dim=1)
        target = target.long()
        
        return target

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(pred.view(-1), target.view(-1))
            
        pt = torch.exp(-BCE_loss)
        at = torch.gather(self.alpha, 0, target.view(-1).long())
        focal_loss = (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        
        return focal_loss
########## END ##########

########## START ##########
########## FocalLoss ver 2 ##########
class FocalLoss2(nn.Module):
    def __init__(self, alpha=[0.25, 0.8, 0.8, 0.8], gamma=2.0, reduction='mean'):
        super(FocalLoss2, self).__init__()
        self.alpha = torch.tensor(alpha, device='cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.reduction = reduction
        self.CE = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')
        
    def un_oh_encode(self, target):
        target = target.float()
        target = torch.argmax(target, dim=1)
        target = target.long()
        
        return target

    def forward(self, pred, target):
        pred = pred.float()
        
        target_index = torch.argmax(target, dim=1).long()
        
        CE_loss = self.CE(pred, target_index)

        sm = torch.softmax(pred - torch.max(pred, dim=1, keepdim=True)[0], dim=1)
        pt = torch.sum(sm * target, dim=1)
        focal_loss = (1-pt)**self.gamma * CE_loss
        
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        
        return focal_loss
########## END ##########

########## START ##########
########## Dice Loss but all at once ##########
class DiceLossTest(nn.Module):
    def __init__(self, mode='batch'):
        super(DiceLossTest, self).__init__()
    
    def forward(self, pred, target, smooth=1e-5):
        pred = pred.float()
        pred = torch.softmax(pred - torch.max(pred, dim=1, keepdim=True)[0], dim=1)
        target = target.float()

        a = pred[:,1:,:,:].reshape(-1)
        b = target[:,1:,:,:].reshape(-1)
        
        inter = (a * b).sum()
        union = a.sum() + b.sum()
        if union == 0:
            score = torch.tensor(1, device='cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            score = (2. * inter + smooth) / (union + smooth)
        diceLoss = (1-score)
        
        return diceLoss
########## END ##########

########## START ##########
########## FocalDice but Dice Loss is all at once ##########
class FocalDiceTest(nn.Module):
    def __init__(self, alpha=[0.25, 0.8, 0.8, 0.8], gamma=2.0, reduction='mean'):
        super(FocalDiceTest, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.focal = FocalLoss(self.alpha, self.gamma, self.reduction)
        self.dice = DiceLossTest()

    def forward(self, pred, target):
        loss = self.focal(pred, target) + self.dice(pred, target)
        return loss
########## END ##########

########## START ##########
########## FocalDice2 ##########
class FocalDice2(nn.Module):
    def __init__(self, alpha=[0.25, 0.8, 0.8, 0.8], gamma=2.0, reduction='mean', s1=1, s2=1):
        super(FocalDice2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.focal = FocalLoss2(self.alpha, self.gamma, self.reduction)
        self.dice = DiceLossTest()
        self.s1 = s1
        self.s2 = s2

    def forward(self, pred, target):
        loss = self.s1*self.focal(pred, target) + self.s2*self.dice(pred, target)
        return loss
########## END ##########

########## START ##########
########## FD ########## (UNUSED)
# class FD(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
#         super(FD, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#         self.focal = FocalLossTest(self.alpha, self.gamma, self.reduction)
#         self.dice = DiceLoss()

#     def forward(self, pred, target):
#         loss = self.focal(pred, target) + self.dice(pred, target)
#         return loss
########## END ##########
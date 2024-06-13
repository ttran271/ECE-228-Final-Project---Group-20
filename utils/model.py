import torch
import torch.nn as nn
import torch.nn.functional as F

#################### READY ####################
########## START ##########
########## UNet ##########
class UNET_Block(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_p):
        super(UNET_Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv1 = nn.Conv2d(in_channel, out_channel, (3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, (3,3), stride=1, padding=1)
        self.dropout = nn.Dropout2d(dropout_p)
        self.relu = nn.ReLU()

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        out = self.dropout(x)
        
        return out

class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        self.B1 = UNET_Block(in_channel=2, out_channel=32, dropout_p=0.1)
        self.B2 = UNET_Block(in_channel=32, out_channel=64, dropout_p=0.1)
        self.B3 = UNET_Block(in_channel=64, out_channel=128, dropout_p=0.1)
        self.B4 = UNET_Block(in_channel=128, out_channel=256, dropout_p=0.1)
        self.B5 = UNET_Block(in_channel=256, out_channel=512, dropout_p=0.1)
        self.B6 = UNET_Block(in_channel=512, out_channel=256, dropout_p=0.1)
        self.B7 = UNET_Block(in_channel=256, out_channel=128, dropout_p=0.1)
        self.B8 = UNET_Block(in_channel=128, out_channel=64, dropout_p=0.1)
        self.B9 = UNET_Block(in_channel=64, out_channel=32, dropout_p=0.1)

        self.upconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2,2), stride=2)
        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2,2), stride=2)
        self.upconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2,2), stride=2)
        self.upconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2,2), stride=2)
        self.mp = nn.MaxPool2d((2,2))

        self.out = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=(1,1))
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = torch.squeeze(x, dim=0)

        # Encoding path
        x1 = self.B1(x)
        x1_down = self.mp(x1)
        x2 = self.B2(x1_down)
        x2_down = self.mp(x2)
        x3 = self.B3(x2_down)
        x3_down = self.mp(x3)
        x4 = self.B4(x3_down)
        x4_down = self.mp(x4)
        x5 = self.B5(x4_down)
        
        # Decoding path
        x5_up = self.upconv1(x5)
        merge1 = torch.cat((x5_up, x4), dim=1)
        y1 = self.B6(merge1)
        y1_up = self.upconv2(y1)

        merge2 = torch.cat((y1_up, x3), dim=1)
        y2 = self.B7(merge2)
        y2_up = self.upconv3(y2)
        
        merge3 = torch.cat((y2_up, x2), dim=1)
        y3 = self.B8(merge3)
        y3_up = self.upconv4(y3)

        merge4 = torch.cat((y3_up, x1), dim=1)
        y4 = self.B9(merge4)
        out = self.out(y4)

        return out
########## END ##########

########## START ##########
########## Residual UNet ##########
class UNET_Skip_Block(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_p, dim_fix=True, batchnorm=True):
        super(UNET_Skip_Block, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.batchnorm = batchnorm

        self.conv1 = nn.Conv2d(in_channel, out_channel, (3,3), stride=1, padding=1)
        self.bn1 = nn.InstanceNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, (3,3), stride=1, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channel)
        self.dropout = nn.Dropout2d(dropout_p)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channel, out_channel, (1,1), stride=1)
        self.bn3 = nn.InstanceNorm2d(out_channel)
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        res = x.clone()
        x = self.conv1(x)
        if self.batchnorm:
            x = self.bn1(x)
        x = self.dropout(self.relu(x))
        x = self.conv2(x)
        if self.batchnorm:
            x = self.bn2(x)
        res = self.conv3(res)
        if self.batchnorm:
            out = self.dropout(self.relu(self.bn3(res) + x))
        else:
            out = self.dropout(self.relu(res + x))
        
        return out

class R_UNET(nn.Module):
    def __init__(self, c=2, start_c=32, batchnorm=True):
        super(R_UNET, self).__init__()
        self.c = c
        self.start_c = start_c
        self.batchnorm = batchnorm

        self.E1 = UNET_Skip_Block(in_channel=self.c, out_channel=self.start_c, dropout_p=0.1, batchnorm=self.batchnorm)
        self.E2 = UNET_Skip_Block(in_channel=self.start_c, out_channel=self.start_c*2, dropout_p=0.1, batchnorm=self.batchnorm)
        self.E3 = UNET_Skip_Block(in_channel=self.start_c*2, out_channel=self.start_c*4, dropout_p=0.1, batchnorm=self.batchnorm)
        self.E4 = UNET_Skip_Block(in_channel=self.start_c*4, out_channel=self.start_c*8, dropout_p=0.1, batchnorm=self.batchnorm)
        self.E5 = UNET_Skip_Block(in_channel=self.start_c*8, out_channel=self.start_c*16, dropout_p=0.1, batchnorm=self.batchnorm)
        self.D1 = UNET_Skip_Block(in_channel=self.start_c*16, out_channel=self.start_c*8, dropout_p=0.1, batchnorm=self.batchnorm)
        self.D2 = UNET_Skip_Block(in_channel=self.start_c*8, out_channel=self.start_c*4, dropout_p=0.1, batchnorm=self.batchnorm)
        self.D3 = UNET_Skip_Block(in_channel=self.start_c*4, out_channel=self.start_c*2, dropout_p=0.1, batchnorm=self.batchnorm)
        self.D4 = UNET_Skip_Block(in_channel=self.start_c*2, out_channel=self.start_c, dropout_p=0.1, batchnorm=self.batchnorm)

        self.upconv1 = nn.ConvTranspose2d(in_channels=self.start_c*16, out_channels=self.start_c*8, kernel_size=(2,2), stride=2)
        self.upconv2 = nn.ConvTranspose2d(in_channels=self.start_c*8, out_channels=self.start_c*4, kernel_size=(2,2), stride=2)
        self.upconv3 = nn.ConvTranspose2d(in_channels=self.start_c*4, out_channels=self.start_c*2, kernel_size=(2,2), stride=2)
        self.upconv4 = nn.ConvTranspose2d(in_channels=self.start_c*2, out_channels=self.start_c, kernel_size=(2,2), stride=2)

        self.downconv1 = nn.Conv2d(in_channels=self.start_c, out_channels=self.start_c, kernel_size=(2,2), stride=2)
        self.downconv2 = nn.Conv2d(in_channels=self.start_c*2, out_channels=self.start_c*2, kernel_size=(2,2), stride=2)
        self.downconv3 = nn.Conv2d(in_channels=self.start_c*4, out_channels=self.start_c*4, kernel_size=(2,2), stride=2)
        self.downconv4 = nn.Conv2d(in_channels=self.start_c*8, out_channels=self.start_c*8, kernel_size=(2,2), stride=2)
        self.mp = nn.MaxPool2d((2,2))

        self.out = nn.Conv2d(in_channels=self.start_c, out_channels=4, kernel_size=(1,1))
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = torch.squeeze(x, dim=0)

        # Encoding path
        x1 = self.E1(x)
        x1_down = self.mp(x1)
        x2 = self.E2(x1_down)
        x2_down = self.mp(x2)
        x3 = self.E3(x2_down)
        x3_down = self.mp(x3)
        x4 = self.E4(x3_down)
        x4_down = self.mp(x4)
        x5 = self.E5(x4_down)
        
        # Decoding path
        x5_up = self.upconv1(x5)
        merge1 = torch.cat((x5_up, x4), dim=1)
        y1 = self.D1(merge1)
        y1_up = self.upconv2(y1)

        merge2 = torch.cat((y1_up, x3), dim=1)
        y2 = self.D2(merge2)
        y2_up = self.upconv3(y2)
            
        merge3 = torch.cat((y2_up, x2), dim=1)
        y3 = self.D3(merge3)
        y3_up = self.upconv4(y3)

        merge4 = torch.cat((y3_up, x1), dim=1)
        y4 = self.D4(merge4)
        out = self.out(y4)

        return out
########## END ##########

########## START ##########
########## Residual UNet (Pre-activated block) ##########
class Pre_Block(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_p, dim_fix=True, norm=True):
        super(Pre_Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.norm = norm

        self.conv1 = nn.Conv2d(in_channel, out_channel, (3,3), stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(in_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, (3,3), stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(out_channel)
        self.dropout = nn.Dropout2d(dropout_p)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channel, out_channel, (1,1), stride=1)
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        res = x.clone()
        if self.norm:
            x = self.in1(x)
        x = self.relu(x)
        x = self.dropout(self.conv1(x))
        if self.norm:
            x = self.in2(x)
        x = self.relu(x)
        x = self.dropout(self.conv2(x))
        out = self.conv3(res) + x

        return out

class PR_UNET(nn.Module):
    def __init__(self, c=2, start_c=32, norm=True):
        super(PR_UNET, self).__init__()
        self.c = c
        self.start_c = start_c
        self.norm = norm

        self.E1 = Pre_Block(in_channel=self.c, out_channel=self.start_c, dropout_p=0.1, norm=self.norm)
        self.E2 = Pre_Block(in_channel=self.start_c, out_channel=self.start_c*2, dropout_p=0.1, norm=self.norm)
        self.E3 = Pre_Block(in_channel=self.start_c*2, out_channel=self.start_c*4, dropout_p=0.1, norm=self.norm)
        self.E4 = Pre_Block(in_channel=self.start_c*4, out_channel=self.start_c*8, dropout_p=0.1, norm=self.norm)
        self.E5 = Pre_Block(in_channel=self.start_c*8, out_channel=self.start_c*16, dropout_p=0.1, norm=self.norm)
        self.D1 = Pre_Block(in_channel=self.start_c*16, out_channel=self.start_c*8, dropout_p=0.1, norm=self.norm)
        self.D2 = Pre_Block(in_channel=self.start_c*8, out_channel=self.start_c*4, dropout_p=0.1, norm=self.norm)
        self.D3 = Pre_Block(in_channel=self.start_c*4, out_channel=self.start_c*2, dropout_p=0.1, norm=self.norm)
        self.D4 = Pre_Block(in_channel=self.start_c*2, out_channel=self.start_c, dropout_p=0.1, norm=self.norm)

        self.upconv1 = nn.ConvTranspose2d(in_channels=self.start_c*16, out_channels=self.start_c*8, kernel_size=(2,2), stride=2)
        self.upconv2 = nn.ConvTranspose2d(in_channels=self.start_c*8, out_channels=self.start_c*4, kernel_size=(2,2), stride=2)
        self.upconv3 = nn.ConvTranspose2d(in_channels=self.start_c*4, out_channels=self.start_c*2, kernel_size=(2,2), stride=2)
        self.upconv4 = nn.ConvTranspose2d(in_channels=self.start_c*2, out_channels=self.start_c, kernel_size=(2,2), stride=2)

        self.mp = nn.MaxPool2d((2,2))

        self.out = nn.Conv2d(in_channels=self.start_c, out_channels=4, kernel_size=(1,1))
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = torch.squeeze(x, dim=0)

        # Encoding path
        x1 = self.E1(x)
        x1_down = self.mp(x1)
        x2 = self.E2(x1_down)
        x2_down = self.mp(x2)
        x3 = self.E3(x2_down)
        x3_down = self.mp(x3)
        x4 = self.E4(x3_down)
        x4_down = self.mp(x4)
        x5 = self.E5(x4_down)
        
        # Decoding path
        x5_up = self.upconv1(x5)
        merge1 = torch.cat((x5_up, x4), dim=1)
        y1 = self.D1(merge1)
        y1_up = self.upconv2(y1)

        merge2 = torch.cat((y1_up, x3), dim=1)
        y2 = self.D2(merge2)
        y2_up = self.upconv3(y2)
            
        merge3 = torch.cat((y2_up, x2), dim=1)
        y3 = self.D3(merge3)
        y3_up = self.upconv4(y3)

        merge4 = torch.cat((y3_up, x1), dim=1)
        y4 = self.D4(merge4)
        out = self.out(y4)

        return out
########## END ##########

########## START ##########
########## Deep Residual UNet ########## (UNUSED)
# class DR_UNET(nn.Module):
#     def __init__(self):
#         super(DR_UNET, self).__init__()
#         # Encoding layers
#         self.E1 = UNET_Skip_Block(in_channel=2, out_channel=32, dropout_p=0.1)
#         self.E2 = UNET_Skip_Block(in_channel=32, out_channel=64, dropout_p=0.1)
#         self.E3 = UNET_Skip_Block(in_channel=64, out_channel=128, dropout_p=0.1)
#         self.E4 = UNET_Skip_Block(in_channel=128, out_channel=256, dropout_p=0.1)
#         self.E5 = UNET_Skip_Block(in_channel=256, out_channel=512, dropout_p=0.1)
#         self.E6 = UNET_Skip_Block(in_channel=512, out_channel=1024, dropout_p=0.1)
#         self.E7 = UNET_Skip_Block(in_channel=1024, out_channel=2048, dropout_p=0.1)

#         # Decoding layers
#         self.D1 = UNET_Skip_Block(in_channel=2048, out_channel=1024, dropout_p=0.1)
#         self.D2 = UNET_Skip_Block(in_channel=1024, out_channel=512, dropout_p=0.1)
#         self.D3 = UNET_Skip_Block(in_channel=512, out_channel=256, dropout_p=0.1)
#         self.D4 = UNET_Skip_Block(in_channel=256, out_channel=128, dropout_p=0.1)
#         self.D5 = UNET_Skip_Block(in_channel=128, out_channel=64, dropout_p=0.1)
#         self.D6 = UNET_Skip_Block(in_channel=64, out_channel=32, dropout_p=0.1)

#         self.upconv1 = nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=(2,2), stride=2)
#         self.upconv2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(2,2), stride=2)
#         self.upconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2,2), stride=2)
#         self.upconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2,2), stride=2)
#         self.upconv5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2,2), stride=2)
#         self.upconv6 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2,2), stride=2)

#         self.out = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=(1,1))
#         self.weight_init()

#     def weight_init(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)

#         # Encoding path
#         x1 = self.E1(x)
#         x1_down = F.max_pool2d(x1, (2,2))
#         x2 = self.E2(x1_down)
#         x2_down = F.max_pool2d(x2, (2,2))
#         x3 = self.E3(x2_down)
#         x3_down = F.max_pool2d(x3, (2,2))
#         x4 = self.E4(x3_down)
#         x4_down = F.max_pool2d(x4, (2,2))
#         x5 = self.E5(x4_down)
#         x5_down = F.max_pool2d(x5, (2,2))
#         x6 = self.E6(x5_down)
#         x6_down = F.max_pool2d(x6, (2,2))
#         x7 = self.E7(x6_down)
        
#         # Decoding path
#         x7_up = self.upconv1(x7)

#         merge1 = torch.cat((x7_up, x6), dim=1)
#         y1 = self.D1(merge1)
#         y1_up = self.upconv2(y1)

#         merge2 = torch.cat((y1_up, x5), dim=1)
#         y2 = self.D2(merge2)
#         y2_up = self.upconv3(y2)

#         merge3 = torch.cat((y2_up, x4), dim=1)
#         y3 = self.D3(merge3)
#         y3_up = self.upconv4(y3)

#         merge4 = torch.cat((y3_up, x3), dim=1)
#         y4 = self.D4(merge4)
#         y4_up = self.upconv5(y4)

#         merge5 = torch.cat((y4_up, x2), dim=1)
#         y5 = self.D5(merge5)
#         y5_up = self.upconv6(y5)

#         merge6 = torch.cat((y5_up, x1), dim=1)
#         y6 = self.D6(merge6)
#         out = self.out(y6)

#         return out
########## END ##########

########## START ##########
########## ResUNet (BatchNorm) ##########
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, (3,3), stride=stride, padding=1)
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, (3,3), stride=1, padding=1)
        self.fix = nn.Conv2d(self.in_channel, self.out_channel, (1,1), stride=stride)
        self.relu = nn.ReLU()
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        resi = x.clone()
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)
        resi = self.fix(resi)
        out = x + resi
        return out

class ResUNET(nn.Module):
    def __init__(self, c=2, start_c=32):
        super(ResUNET, self).__init__()
        self.c = c
        self.start_c = start_c
        
        self.conv1 = nn.Conv2d(self.c, self.start_c, (3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.start_c, self.start_c, (3,3), stride=1, padding=1)
        self.fix = nn.Conv2d(self.c, self.start_c, (1,1), stride=1)
        self.bn = nn.BatchNorm2d(self.start_c)
        self.relu = nn.ReLU()

        # Encoder blocks
        self.E1 = ResBlock(self.start_c, self.start_c*2, stride=2)
        self.E2 = ResBlock(self.start_c*2, self.start_c*4, stride=2)
        self.E3 = ResBlock(self.start_c*4, self.start_c*8, stride=2)
        self.E4 = ResBlock(self.start_c*8, self.start_c*16, stride=2)

        #Decoder blocks
        self.D1 = ResBlock(self.start_c*16, self.start_c*8)
        self.D2 = ResBlock(self.start_c*8, self.start_c*4)
        self.D3 = ResBlock(self.start_c*4, self.start_c*2)
        self.D4 = ResBlock(self.start_c*2, self.start_c)

        #Up-conv + out
        self.upconv1 = nn.ConvTranspose2d(in_channels=self.start_c*16, out_channels=self.start_c*8, kernel_size=(2,2), stride=2)
        self.upconv2 = nn.ConvTranspose2d(in_channels=self.start_c*8, out_channels=self.start_c*4, kernel_size=(2,2), stride=2)
        self.upconv3 = nn.ConvTranspose2d(in_channels=self.start_c*4, out_channels=self.start_c*2, kernel_size=(2,2), stride=2)
        self.upconv4 = nn.ConvTranspose2d(in_channels=self.start_c*2, out_channels=self.start_c, kernel_size=(2,2), stride=2)
        self.out = nn.Conv2d(self.start_c, 4, (1,1), stride=1)

    def forward(self, x):
        x = x.squeeze(0)
        resi = x.clone()
        x = self.conv1(x)
        x = self.relu(self.bn(x))
        x = self.conv2(x)
        x1 = self.fix(resi) + x

        # Encoding path
        x2 = self.E1(x1)
        x3 = self.E2(x2)
        x4 = self.E3(x3)
        x5 = self.E4(x4)
        
        # Decoding path
        x5_up = self.upconv1(x5)
        m1 = torch.cat((x5_up, x4), dim=1)
        y1 = self.D1(m1)

        y1_up = self.upconv2(y1)
        m2 = torch.cat((y1_up, x3), dim=1)
        y2 = self.D2(m2)
        
        y2_up = self.upconv3(y2)
        m3 = torch.cat((y2_up, x2), dim=1)
        y3 = self.D3(m3)

        y3_up = self.upconv4(y3)
        m4 = torch.cat((y3_up, x1), dim=1)
        y4 = self.D4(m4)

        out = self.out(y4)

        return out
########## END ##########
    
#################### TESTING ####################
########## START ##########
########## ResUNET (InstanceNorm) ##########
class ResBlock2(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock2, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.bn1 = nn.InstanceNorm2d(in_channel)
        self.bn2 = nn.InstanceNorm2d(out_channel)
        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, (3,3), stride=stride, padding=1)
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, (3,3), stride=1, padding=1)
        self.fix = nn.Conv2d(self.in_channel, self.out_channel, (1,1), stride=stride)
        self.relu = nn.ReLU()
        
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        resi = x.clone()
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)
        resi = self.fix(resi)
        out = x + resi
        return out

class ResUNET_Test(nn.Module):
    def __init__(self, c=2, start_c=32):
        super(ResUNET_Test, self).__init__()
        self.c = c
        self.start_c = start_c
        
        self.conv1 = nn.Conv2d(self.c, self.start_c, (3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.start_c, self.start_c, (3,3), stride=1, padding=1)
        self.fix = nn.Conv2d(self.c, self.start_c, (1,1), stride=1)
        self.bn = nn.InstanceNorm2d(self.start_c)
        self.relu = nn.ReLU()

        # Encoder blocks
        self.E1 = ResBlock(self.start_c, self.start_c*2, stride=2)
        self.E2 = ResBlock(self.start_c*2, self.start_c*4, stride=2)
        self.E3 = ResBlock(self.start_c*4, self.start_c*8, stride=2)
        self.E4 = ResBlock(self.start_c*8, self.start_c*16, stride=2)

        #Decoder blocks
        self.D1 = ResBlock(self.start_c*16, self.start_c*8)
        self.D2 = ResBlock(self.start_c*8, self.start_c*4)
        self.D3 = ResBlock(self.start_c*4, self.start_c*2)
        self.D4 = ResBlock(self.start_c*2, self.start_c)

        #Up-conv + out
        self.upconv1 = nn.ConvTranspose2d(in_channels=self.start_c*16, out_channels=self.start_c*8, kernel_size=(2,2), stride=2)
        self.upconv2 = nn.ConvTranspose2d(in_channels=self.start_c*8, out_channels=self.start_c*4, kernel_size=(2,2), stride=2)
        self.upconv3 = nn.ConvTranspose2d(in_channels=self.start_c*4, out_channels=self.start_c*2, kernel_size=(2,2), stride=2)
        self.upconv4 = nn.ConvTranspose2d(in_channels=self.start_c*2, out_channels=self.start_c, kernel_size=(2,2), stride=2)
        self.out = nn.Conv2d(self.start_c, 4, (1,1), stride=1)

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = x.squeeze(0)
        resi = x.clone()
        x = self.conv1(x)
        x = self.relu(self.bn(x))
        x = self.conv2(x)
        x1 = self.fix(resi) + x

        # Encoding path
        x2 = self.E1(x1)
        x3 = self.E2(x2)
        x4 = self.E3(x3)
        x5 = self.E4(x4)
        
        # Decoding path
        x5_up = self.upconv1(x5)
        m1 = torch.cat((x5_up, x4), dim=1)
        y1 = self.D1(m1)

        y1_up = self.upconv2(y1)
        m2 = torch.cat((y1_up, x3), dim=1)
        y2 = self.D2(m2)
        
        y2_up = self.upconv3(y2)
        m3 = torch.cat((y2_up, x2), dim=1)
        y3 = self.D3(m3)

        y3_up = self.upconv4(y3)
        m4 = torch.cat((y3_up, x1), dim=1)
        y4 = self.D4(m4)

        out = self.out(y4)

        return out
########## END ##########
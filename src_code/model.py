import torch
##### creates a a UNET with three DownConvolutions and three UpConvolutions.
##### Uses 64, 128, 256 and 512 filters for both, Down and UpConvolutions and skip-connections.

class DoubleConv(torch.nn.Module):
    """
    Helper class to implement a ReLU activated double convolution with padding and a kernel size of 3x3
    """
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.step = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU())
        
    def forward(self, X):
        return self.step(X)
    
class Unet(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        ##### decoding layers
        self.layer1=DoubleConv(1,64)
        self.layer2=DoubleConv(64,128)
        self.layer3=DoubleConv(128,256)
        self.layer4=DoubleConv(256,512)
        ##### encoding layers with skips  
        self.layer5=DoubleConv(512+256,256)
        self.layer6=DoubleConv(256+128,128)
        self.layer7=DoubleConv(128+64,64)
        self.layer8=torch.nn.Conv2d(64,1,1)
        #### 
        self.maxpooling=torch.nn.MaxPool2d(2)
        self.upsample=torch.nn.Upsample(scale_factor=2, mode='bilinear')
    
    #### forward pass
    def forward(self,X):
        #### encoder
        #### before maxpooling
        X1bm=self.layer1(X)
        #### after maxpooling
        X1am=self.maxpooling(X1bm)
        #### before maxpooling
        X2bm=self.layer2(X1am)
        #### after maxpooling
        X2am=self.maxpooling(X2bm)
        #### before maxpooling
        X3bm=self.layer3(X2am)
        #### after maxpooling
        X3am=self.maxpooling(X3bm)
        X4=self.layer4(X3am)
        #### decoder
        X5=self.upsample(X4)
        #### using torch.cat to concatenate with parts coming from the skip connections
        #### matrices assigned before maxpooling 
        X6=self.layer5(torch.cat([X3bm, X5], dim=1))
        X7=self.upsample(X6)
        X8=self.layer6(torch.cat([X2bm, X7], dim=1))
        X9=self.upsample(X8)
        X10=self.layer7(torch.cat([X1bm, X9], dim=1))
        Xout=self.layer8(X10)
        return Xout
    

### Model Testbench
model=Unet()
##### we test a random tensor input of 1x1x256x256, which has the same dimensions with the images in the dataset
xrand=torch.randn(1,1,256,256)
xout=model(xrand)
#### ensure we get 1x1x256x256 at the output
print(xout.shape)
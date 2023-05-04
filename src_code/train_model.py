from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import nibabel as nib
import cv2
from celluloid import Camera
from IPython.display import HTML
from Dataset import LungDataset
from model import Unet

##### CASE Dataset module is not detected when batches are extracted--(Possible BUG)---
#####-We fix it by running the following code:
"""
import sys
projectpath=----Directory_to_code_files----
sys.path.insert(0,projectpath)
"""
#####

######## Dataset Creation #############
######## Creating the train and val dataset and the augmentation pipeline
seq=iaa.Sequential([iaa.Affine(scale=(0.85,1.15),
                               rotate=(-45,45),
                               translate_percent={"x": (-0.15, 0.15),
                                                  "y": (-0.15, 0.15)}),
                    iaa.ElasticTransformation()])

train_path=Path('Preprocessed_Lung_Tumor/train/')
val_path=Path('Preprocessed_Lung_Tumor/val/')
train_dataset=LungDataset(train_path,seq)
val_dataset=LungDataset(val_path,None)

print(f'There are {len(train_dataset)} train images and {len(val_dataset)} val images')

######## Oversampling to tackle strong class imbalance
######## for sufficient training, we need to take sample slices which contain a tumor more often
######## WeightedRandomSampler will be used here
######## create a list containing only the class labels
target_list = [np.any(label).astype(np.int8) for _, label in tqdm(train_dataset)]
target_list = np.array(target_list)
non_tumorousdivtum = np.round((1 - target_list.mean()) / target_list.mean(), 3)
print('clear/tumorous data fraction:', non_tumorousdivtum)

#### create a weight list, to assign the fraction when tumorous label is present, and 1 when non-present
weight_list = np.copy(target_list).astype(np.float16)
weight_list[weight_list==1]=non_tumorousdivtum
weight_list[weight_list==0]=1
#### define the corresponding samples, only for the train loader
sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_list, len(weight_list))  

#### Create the train and val_loaders. 
batch_size=8 ##### large batch sizes would probably cause memoryerrors if using mediocre GPUs
num_workers=4
#### sampler applied only in training data
#### we do not use sfuffle with  samplers
train_loader=torch.utils.data.DataLoader(train_dataset,\
                                         batch_size=batch_size,\
                                             num_workers=num_workers,\
                                                 sampler=sampler)
#### never sample of shuffle validation data
val_loader=torch.utils.data.DataLoader(val_dataset,\
                                         batch_size=batch_size,\
                                             num_workers=num_workers,\
                                                 shuffle=False)
    
#### verify that sampler works by taking a batch from the train loader
#### count the >0 labels
verify_sampler = next(iter(train_loader))  # Take one batch
label_sampler=verify_sampler[1][:,0] ### take the labels (size 16x1x256x256 and squeeze to 16x256x256)
tumorus_in_batch=sum(label_sampler.sum([1, 2]) > 0)   # how many out of the batch size labels 
##### are larger than zero, after summing each 256x256 mask elements (along dimensions 1 and 2)
print(f'In this {batch_size}-size random taken batch, we have a number of {tumorus_in_batch} tumorous cases')
    

######## Define Dice score for later evaluation
class DiceLoss(torch.nn.Module):
    """
    class to compute the Dice Loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):
                
        # Flatten label and prediction tensors
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)
        counter = (pred * mask).sum()  # Numerator       
        denum = pred.sum() + mask.sum() + 1e-8  # Denominator. Add a small number to prevent NANS
        dice =  (2*counter)/denum
        return 1 - dice

######## Define the Segmenetation model using Pytorch LightningModule #############

class LungDetector(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model=Unet()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4)
        #### as loss function we used the cross entropy loss, however dice loss could be tried either
        self.loss_fn=torch.nn.BCEWithLogitsLoss()
        
    def forward(self,data):
      ### including the sigmoid activation function here causes training instability due to saturation close to 1
      ### the sigmoid activation function is evaluated at the evaluation stage, out of this particular class
        return self.model(data)
    
    def training_step(self, batch, batch_idx):
        ct, mask= batch
        mask=mask.float()
        pred=self(ct.float())
        loss=self.loss_fn(pred, mask)      
        self.log('Train Loss', loss)
        if batch_idx%50==0:
            print('training loss:',loss)
            self.log_images(ct.cpu(), pred.cpu(), mask.cpu(), 'Train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        ct, mask= batch
        mask=mask.float()
        pred=self(ct.float())
        loss=self.loss_fn(pred,mask)
        self.log('Val Dice', loss)
        if batch_idx%2==0:
            print('validation loss:',loss)
            self.log_images(ct.cpu(), pred.cpu(), mask.cpu(), 'Val')
        return loss
    
    def log_images(self, ct, pred, mask, name):
        pred=pred>0.5
        
        fig,axis=plt.subplots(1,2)
        axis[0].imshow(ct[0][0], cmap='bone')
        mask_=np.ma.masked_where(mask[0][0]==0, mask[0][0])
        axis[0].imshow(mask_,alpha=0.6)
        axis[1].imshow(ct[0][0], cmap='bone')
        mask_=np.ma.masked_where(pred[0][0]==0, pred[0][0])
        axis[1].imshow(mask_,alpha=0.6)
        self.logger.experiment.add_figure(name, fig, self.global_step)
        
    def configure_optimizers(self):
        return [self.optimizer]

## Instanciating the model
torch.manual_seed(0)
model=LungDetector()

## Creating a checkpoint callback
checkpoint_callback=ModelCheckpoint(monitor='Val Loss', save_top_k=5, mode='min')
## Defining the trainer with a TensorboardLogger.
devices=1
accelerator='gpu'
num_epochs=30
logs_dir='logs'
trainer=pl.Trainer(devices=devices,accelerator=accelerator,logger=TensorBoardLogger(save_dir=logs_dir),\
                   log_every_n_steps=1,callbacks=checkpoint_callback, max_epochs=num_epochs) 

## Train the model
trainer.fit(model,train_loader,val_loader)

######## EVALUATION ##########
#### save weights of the latest checkpoint in the same directory with the code (if using colab, download it)
#### get the latest one

import os
CheckpointDict=Path('logs/checkpoints/')
list_of_files = CheckpointDict.glob('*.ckpt') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
Path2model=latest_file

#### save
trainer.save_checkpoint(latest_file)

#### load weights to the model and evaluate
model=LungDetector.load_from_checkpoint(Path2model,strict=False,map_location=torch.device('cpu'))
model.eval();
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

#### get the predictions for validation data
preds=[]
labels=[]
for slic, label in tqdm(val_dataset):
  slic=torch.tensor(slic).to(device).unsqueeze(0)
  with torch.no_grad():
    pred=torch.nn.sigmoid(model(slic))
  preds.append(pred.cpu().numpy())
  labels.append(label)
preds=np.array(preds)
labels=np.array(labels)

###### take a snapshot of predicted vs GT masks
fig, axis = plt.subplots(1, 2)
axis[0].imshow(preds[9][0][0])
axis[1].imshow(labels[9][0])

###### compute the dice score, regarding the model's predictions and tha GT labels
dice_score = 1-DiceLoss()(torch.from_numpy(preds), torch.from_numpy(labels).unsqueeze(0).float())
print(f"The Val Dice Score is: {dice_score}")
###### Dice score expected to 50% since we deal with small tumours

###### Visualization ######
####### Computing a prediction for a patient and visualize the prediction ######

subject_datapath=Path('Task06_Lung/imagesTs/lung_087.nii')
subject_ct = nib.load(subject_datapath).get_fdata()/3071 #### load data numpy array and standardize it
subject_ct = subject_ct[:,:,30:-30]  # crop
###### resize each slice using OpenCV without employing the mask
preds = []
###### and get the model's predictions on each slice of this particular subject
for i in range(subject_ct.shape[-1]):
    slic = subject_ct[:,:,i]
    slic_res=cv2.resize(slic,(256,256))
    with torch.no_grad():
        pred = torch.nn.sigmoid(model(torch.tensor(slic_res).unsqueeze(0).unsqueeze(0).float().to(device)))[0][0]
        pred = pred > 0.5
    preds.append(pred.cpu())

###### now play the video: scans with masks
fig = plt.figure()
camera = Camera(fig)  # create the camera object from celluloid

for i in range(subject_ct.shape[-1]):
    plt.imshow(subject_ct[:,:,i], cmap="bone")
    mask = np.ma.masked_where(preds[i]==0, preds[i])
    plt.imshow(mask, alpha=0.5, cmap="autumn")
    
    camera.snap()  # Store the current slice
animation = camera.animate()  # create the animation


from IPython.display import HTML
HTML(animation.to_html5_video())

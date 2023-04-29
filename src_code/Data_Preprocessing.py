##Task: Import the necessary libraries
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import cv2
from celluloid import Camera
from IPython.display import HTML
import imgaug.augmenters as iaa

#############Inspection#######################
########## Task: Define the paths to images and labels
path2files='Task06_Lung/'
root=Path(path2files+'imagesTr')
label=Path(path2files+'labelsTr')

###### Task: Load a sample NIfTI and its corresponding label mask

#### helper function
def change_img_to_label_path(path):
    parts=list(path.parts)
    parts[parts.index('imagesTr')]='labelsTr'
    return Path(*parts)

sample_path=list(root.glob('lung*'))[1]
sample_path_label=change_img_to_label_path(sample_path)

##### check existence of sample files
print(sample_path)
print(sample_path_label)
#####

##### load NIftI (sample)
data=nib.load(sample_path)
label=nib.load(sample_path_label)

ct=data.get_fdata()
mask=label.get_fdata().astype(np.uint8)

##### check orientation
print(nib.aff2axcodes(data.affine))

##### Task: Inspect the loaded data with overlaid Ground Truth tumor segmentation

#### figure an example, along with mask
slicerz=140
plt.imshow(ct[:,:,slicerz], cmap='bone')
mask_=np.ma.masked_where(mask[:,:,slicerz]==0, mask[:,:,slicerz])
plt.imshow(mask_, alpha=0.5,cmap='cool')

####video of all slices
fig=plt.figure()
camera=Camera(fig)

for slic in range(ct.shape[2]): ### along LAST axis (height)
    plt.imshow(ct[:,:,slic],cmap='bone')
    mask_=np.ma.masked_where(mask[:,:,slic]==0, mask[:,:,slic])
    plt.imshow(mask_,alpha=0.5,cmap='cool')
    camera.snap()
animation=camera.animate()

###############Preprocessing##########################
### Normalize 
#### CT images have a fixed range from -1000 to 3071
def  normalize(full_volume):
    normalized=full_volume/3071
    return normalized

#### LOOP ALL image files to normalize, crop, take 2D slices and resize them
all_files=list(root.glob('lung*'))
print(len(all_files))
#### at first define the root folder where preprocessed data is going to be stored
save_root=Path('Preprocessed_Lung_Tumor/')

#### run the loop 
for counter, path_to_ct_data in enumerate(tqdm(all_files)):
    path_to_label=change_img_to_label_path(path_to_ct_data)
    ct=nib.load(path_to_ct_data)
    ### make sure all scans have the same orientation
    assert nib.aff2axcodes(ct.affine)==('L','A','S')
    ### now all patients' data is temporarily passed to ct_data, label_data
    ct_data=ct.get_fdata()
    label_data=nib.load(path_to_label).get_fdata().astype(np.uint8)
    #### cut slices: skip the first and last 30 ones (lower abdomen and neck, keep lungs' window)
    ct_data=ct_data[:,:,30:-30]
    label_data=label_data[:,:,30:-30]
    ##### normalize
    normalized_ct_data=normalize(ct_data)
    train_nopatients=len(all_files)-6 ### last 6 for validation data
    #### distinguish training and validation data in different directories
    if counter<train_nopatients:
        current_path=save_root/'train'/str(counter)
    else:
        current_path=save_root/'val'/str(counter)
    ##### now loop each patient's ct_data slices and corresponding masks to resize and store them
    for slic in range(normalized_ct_data.shape[-1]):
        sliced=normalized_ct_data[:,:,slic]
        mask=label_data[:,:,slic]
        ##### resize!!!!
        sliced_resized=cv2.resize(sliced,(256,256))
        mask_resized=cv2.resize(mask,(256,256),interpolation = cv2.INTER_NEAREST)
        slice_path=current_path/'data'
        mask_path=current_path/'masks'
        slice_path.mkdir(parents=True, exist_ok=True)
        mask_path.mkdir(parents=True,exist_ok=True)
        np.save(slice_path/str(slic),sliced_resized) 
        np.save(mask_path/str(slic),mask_resized)
        

path=Path('Preprocessed_Lung_Tumor/train/19')
file='200.npy'
slic_data=np.load(path/'data'/file)
mask=np.load(path/'masks'/file)

plt.figure()
plt.imshow(slic_data,cmap='bone')
mask_=np.ma.masked_where(mask==0,mask)
plt.imshow(mask_, alpha=0.5,cmap='cool')
print(slic_data.min(), slic_data.max())

####Task: Test the dataset by showing the same (tumorous) slice 9 times. 
#### Make sure that the label mask is augmented correctly 
#### augementation pipeline
seq = iaa.Sequential([
    iaa.Affine(scale=(0.85, 1.15), # zoom in or out
               rotate=(-45, 45)),  # rotate up to 45 degrees
    iaa.ElasticTransformation()
                ])

#### create dataset
from Dataset import LungDataset
path=Path('Preprocessed_Lung_Tumor/train/')
dataset=LungDataset(path,seq)


## good enough, just try to find an im where tumor exists
fig, axis = plt.subplots(3, 3, figsize=(9, 9))
im=120
for i in range(3):
    for j in range(3):
        slice, mask = dataset[im]
        mask_ = np.ma.masked_where(mask==0, mask)
        axis[i][j].imshow(slice[0], cmap="bone")
        axis[i][j].imshow(mask_[0], cmap="autumn")
        axis[i][j].axis("off")
fig.suptitle("Sample augmentations")
plt.tight_layout()

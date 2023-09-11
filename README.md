# CARE: A Large Scale CT Image Dataset and Clinical Applicable Benchmark Model for Rectal Cancer Segmentation

### Overview

We provide our implementation of U-SAM in **u-sam.py**, **backbone.py** and  **segment_anything**. The **dataloaders** of **CARE** and **WORD** are also available in **dataset**.

### U-SAM

In **u-sam.py**, we introduced a novel U-shaped architecture to the SAM while retaining its promptable paradigm. Additionally, we provide codes for both the training and evaluating process as well. 

### Backbone

In **backbone.py**, we implemented U-SAM's convolutional upsampling and downsampling modules with minimal modification to the original UNet model. 

### Segment_anything

In **segment_anything**, we made essential modifications to integrate the original SAM to our proposed U-shaped framework. 

### Dataset

in **dataset**, we offer the dataloaders of CARE and WORD. In each dataloader, we illustrate how to load from the target dataset, perform data augmentation and generate valid prompt. 

## Pre-trained Weights

We utilized the SAM-ViT-B in our model, the pre-trained weights of which are supposed to be placed in the folder **weight**. Due to the file size limit, we can't attach the weights to the supplement  materials. So we alternatively provide open access to the official pre-trained weights of SAM-ViT-B as follows: 

https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth. 

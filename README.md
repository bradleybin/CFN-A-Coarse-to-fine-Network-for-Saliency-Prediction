# CFN:A Coarse-to-fine Network for Saliency Prediction Guidance

This Python script demonstrates the usage of our proposed CFN for saliency prediction. 

## Dependencies
- Python 3.6
- PyTorch 1.0.0
- torchvision
- numpy
- PIL
- scipy

## Source Code

- demo.py:                             demonstrates the usage of this package. 
- models.py:                            Fine perceiving network. 
- NASNET_conv.py:                       Coarse perceiving network.
- utils.py:                             Padding and data-download functions.

## Usage

We built two different versions of our model: one based on the SALICON and the other based on the MIT1003. Thus, the input size of the first version should be 640\*480 and this version is applied to SALICON. The input size of the second one is unfixed and this version is applied to MIT1003, MIT300, SUN500, OSIE, and Toronto.

### 1. Download Pretrained Models

Download one of the following pretrained models and save it in the checkpoint folder:
* First version based on SALICON : **(https://github.com/marcellacornia/sam/releases/download/1.0/sam-vgg_salicon_weights.pkl)**
* Second version based on MIT1003 : **(https://github.com/marcellacornia/sam/releases/download/1.0/sam-vgg_salicon_weights.pkl)**

### 2. Predict Saliency Maps

#### To compute saliency maps for the SALICON dataset:
```
python demo.py salicon path/to/images/folder/
```
where ```"path/to/images/folder/"``` is the path of a folder containing the images for which you want to calculate the saliency maps.

For example
```
python demo.py salicon salicon_images
```
The predicted saliency maps will in the output folder


#### To compute saliency maps for the MIT1003, MIT300, SUN500, OSIE, and Toronto datasets:
```
python demo.py mit path/to/images/folder/
```
where ```"path/to/images/folder/"``` is the path of a folder containing the images for which you want to calculate the saliency maps.  

For example
```
python demo.py mit mit_images
```
The predicted saliency maps will in the output folder




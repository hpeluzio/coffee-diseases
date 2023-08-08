# Coffee diseases classification using Vision in Transform

The whole codebase is implemented in Pytorch, which makes it easier for you to tweak and experiment.

Using the repository is straightforward - all you need to do is run the `train_coffee.py` script with different arguments, depending on the model and training parameters you'd like to use.

The dataset used for this work came from the following work: BrACoL, RoCoLe and LiCoLe.

The article that inspired this work was <a target=_blank href="https://link.springer.com/article/10.1007/s00138-022-01277-y">Automated diagnosis of diverse coffee leaf images through a stage-wise aggregated triple deep convolutional neural network</a>.

# Usage example
`python train_coffee.py` # vit-patchsize-4

`python train_coffee.py  --size 48` # vit-patchsize-4-imsize-48

`python train_coffee.py --patch 2` # vit-patchsize-2

`python train_coffee.py --net vit_small --n_epochs 400` # vit-small

`python train_coffee.py --net vit_timm` # train with pretrained vit

`python train_coffee.py --net convmixer --n_epochs 400` # train with convmixer

`python train_coffee.py --net mlpmixer --n_epochs 500 --lr 1e-3`

`python train_coffee.py --net cait --n_epochs 200` # train with cait

`python train_coffee.py --net swin --n_epochs 400` # train with SwinTransformers

`python train_coffee.py --net res18` # resnet18+randaug

## DATASETS

<p>The dataset used for this work came from the following works:</p>

**Please cite or credit their work when using it!** 

**RoCoLe** 
<p>Parraga-Alava, Jorge; Cusme, Kevin; Loor, Angélica; Santander, Esneider (2019), 
<b>“RoCoLe: A robusta coffee leaf images dataset ”</b>
<i>Mendeley Data</i>, V2, doi: <a target=_blank href="http://dx.doi.org/10.17632/c5yvn32dzg.2">10.17632/c5yvn32dzg.2</a></p>

Inclusion: 
- Healthy
- Coffee Leaf Rust (CLR)
- Red Spider Mites (RSM) 

**BrACoL** 
<p>Krohling, Renato; esgario, José; Ventura, Jose A. (2019),
<b>“BRACOL - A Brazilian Arabica Coffee Leaf images dataset to identification and quantification of coffee diseases and pests”</b>
<i>Mendeley Data</i>, V1, doi: <a target=_blank href="http://dx.doi.org/10.17632/yy2k5y8mxg.1">10.17632/yy2k5y8mxg.1</a></p>

<p>Esgario, J. G., Krohling, R. A., & Ventura, J. A. (2020) 
<b>"Deep learning for classification and severity estimation of coffee leaf biotic stress"</b>
<i>Computers and Electronics in Agriculture</i>
169, 105162. doi:<a href="https://doi.org/10.1016/j.compag.2019.105162">10.1016/j.compag.2019.105162</a></p>

Inclusion: 
- Healthy
- CLR
- Cercospora Leaf Spots (CLS)
- Phoma Leaf Spots (PLS)
- Coffee Leaf Miner (CLM)

**LiCoLe**
<p>Montalbo, Francis Jesmar Perez; Hernandez, Alexander Arsenio (2020) 
<b>"Classifying Barako coffee leaf diseases using deep convolutional models"</b>
<i>International Journal of Advances in Intelligent Informatics (IJAIN)</i>
[S.l.], v. 6, n. 2, p. 197-209, july 2020. ISSN 2548-3161. doi: <a href="https://doi.org/10.26555/ijain.v6i2.495">10.26555/ijain.v6i2.495</a></p>

<p>Montalbo, Francis Jesmar Perez
<b>"An Optimized Classification Model for Coffea Liberica Disease using Deep Convolutional Neural Networks"</b>
<i>n Proc. of the 2020 16th IEEE International Colloquium on Signal Processing & Its Applications (CSPA),</i> 
  Langkawi, Malaysia, 2020, pp. 213-218, doi: <a href="https://ieeexplore.ieee.org/document/9068683">10.1109/CSPA48992.2020.9068683</a>.</p>

Inclusion: 
- Healthy
- CLR
- Sooty Molds (SM)

**:heavy_exclamation_mark: For the readily prepared dataset used in this work refer to this link (OPTIONAL) 🠊 <a target=blank_ href="https://drive.google.com/drive/u/1/folders/1FyTnzfz0iLiiRMVWumEaoyFkX2YOHWz3">Google Drive Prepared Dataset<a/>** 
  
`PREPARED DATASET: (7 GB)`

**NOTE: The following credits for the datasets still goes to their appropriate owners and collectors.** 
***Please remember to cite their work when using their respective datasets.***

## Environment Setup

**Make sure to create a new virtual environment preferably in Anaconda**

https://www.anaconda.com/

https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

Activate and access the folder `dataset/`.

https://developer.nvidia.com/cuda-toolkit

https://developer.nvidia.com/rdp/cudnn-archive

Afterwards, simply enter the command in the conda CLI `pip install -r requirements.txt`


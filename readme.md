# Face Sketch to Image Generation using GAN

An image generation system using GAN to turn face sketches into realistic photos

## Install requirements
conda create -n gan python=3.6 -y
conda activate gan
python -m pip install --upgrade pip
pip install -r requirements.txt

## Keras-contrib installation
- git clone https://www.github.com/keras-team/keras-contrib.git
- cd keras-contrib
- python setup.py install

Or you can refer to this link https://medium.com/@kegui/how-to-install-keras-contrib-7b75334ab742
pip install -r requirement.txt

## Install cuda=10.0
conda install cudatoolkit=10.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/linux-64/ 

## Install cudnn=7.6.5
conda install conda-forge::cudnn 

## Data Augmentation
First of all, you need to do data augmentation using this [notebook](https://github.com/Malikanhar/Sketch-to-Image/blob/master/Data%20Augmentation.ipynb)

## Start Training
Start training GAN model with this [notebook](https://github.com/Malikanhar/Sketch-to-Image/blob/master/ContextualGAN.ipynb)

## Performance Measurement
Calculate SSIM (Structural Similarity Index) and Verification Accuracy (L2-norm) using this [notebook](https://github.com/Malikanhar/Sketch-to-Image/blob/master/Compute%20SSIM%20and%20L2-norm.ipynb)

## Testing
Generate single image with this [notebook](https://github.com/Malikanhar/Face-Sketch-to-Image-Generation-using-GAN/blob/master/Predict%20Image.ipynb)

## References
<a id="1">[1]</a> 
X. Wang and X. Tang. (2009).
Face Photo-Sketch Synthesis and Recognition. 
IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 31(11), 1955-1967.

<a id="2">[2]</a>
W. Zhang, X. Wang and X. Tang. (2011).
Coupled Information-Theoretic Encoding for Face Photo-Sketch Recognition.
Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

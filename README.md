# Bird-Species-Classification-Using-Transfer-Learning

This project implements bird species classification using transfer learning (VGG16bn and ResNet18).

## Dataset  

The dataset contains 12,000 images of 200 bird species. We will be working on a small subset of this dataset with 20 bird species having 743 training images and 372 images for validation.

Caltech-UCSD Birds-200-2011 (CUB-200-2011): [https://sites.google.com/visipedia.org/index](https://sites.google.com/visipedia.org/index)

* Download here: [http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)

This directory contains a folder `CUB_200_2011` with all the images and two files: `train.csv` and `val.csv`. Each line of these files correponds to a sample described by the file path of the image, the bounding box values surrounding the bird, and the respective class label for each species from 0 to 19 (separated by commas). Given the very small size of this subset, we will rely on transfer learning (otherwise we will be facing the curse of dimensionality).

## Testing Environment  

* Pytorch version: `1.0.0`
* CUDA version: `9.0.176`
* Python version: `3.6.8`
* CPU: Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz
* GPU: GeForce GTX 1080 Ti (11172MB GRAM)
* RAM: 32GB

## Usage

1. Clone this repository

```bash
git clone https://github.com/lychengr3x/Bird-Species-Classification-Using-Transfer-Learning.git
```

2. Download dataset

```bash
cd Bird-Species-Classification-Using-Transfer-Learning/dataset
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar xvzf CUB_200_2011.tgz
rm CUB_200_2011.tgz
```

3. Train the model

```bash
cd ../src
python main.py
```

**PS**: Read [`argument.py`](src/argument.py) to see what parameters that you can change.

## Demonstration and tutorial

Please see [`demo.ipynb`](src/demo.ipynb) for demonstration, and [`tutorial.ipynb`](src/tutorial.ipynb) for tutorial.
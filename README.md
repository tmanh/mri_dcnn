# Automated diagnosis of prostate cancer in multi-parametric MRI based on multimodal convolutionalneural networks

This is the implementation of the paper: Automated diagnosis of prostate cancer in multi-parametric MRI based on multimodal convolutionalneural networks, MH Le, 2017.

## Dependencies

```
conda create -n mri python=3.7
conda activate mri
conda install numpy
conda install -c conda-forge nibabel
conda install scikit-image
conda install -c conda-forge scikit-learn
pip install pystackreg
pip install tensorflow-gpu (or tensorflow)
pip install mahotas
```

## Preprocessing

The processing of the paper "Automated diagnosis of prostate cancer in multi-parametric MRI based on multimodal convolutionalneural networks" requires the image registration library called MIRT. I have write a sample [script](./MIRT/preprocess.m) to register all the data. You have to modify or write a simple conversion data code from your npy data to png image to use the code.
## Data augmentation

The data augmentation code is in [utils file](./utils.py) including both non-rigid (data_augmentation_non_rigid) and rigid (data_augmentation_rigid) appoaches. The code is implemented with Tensorflow (v2.4).

## Network implementation

I have implemented all the methods in the paper. If you just wanted to get the best model described in the paper, run the code as follows:

```
train(modality='resnet50', combine_mode='c3', fusion_mode='paper')                                    # to train the network
predict(None, None, None, mode='train', modality='resnet50', combine_mode='c3', fusion_mode='paper')  # to get the merged features and the predictions on train data
predict(None, None, None, mode='test', modality='resnet50', combine_mode='c3', fusion_mode='paper')   # to get the merged features and the predictions on test data
svm_stage_1()                                                                                         # run the svm stage-1 to get the predictions
svm_stage_2_combine_3()                                                                               # run svm stage-2 with combine-3 to get the final results
```

Then run the code by:

```
python main.py
```

The data used in this implementation is just an example, not a real MRI data. You have to modify the code for the actual MRI data. Please find all the ```NOTE``` in the code to modify.

## Directory structures

```
.
├── README.md                                  # this readme
├── data                                       # sample data folder
├── engine.py                                  # code for training and testing
├── googlenet.py                               # implementation of googlenet
├── handcraft.py                               # code to extract mean, std, skewness, kurtosis and haralick features
├── main.py                                    # run code to train or test
├── metrics.py                                 # implementation of the assessments
├── models.py                                  # implementation of the CNN models of the paper
└── utils.py                                   # augmentation code
```

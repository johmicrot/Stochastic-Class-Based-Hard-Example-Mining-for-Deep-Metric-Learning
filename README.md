# Stochastic-Class-Based-Hard-Example-Mining-for-Deep-Metric-Learning (SCHEM)
Implementation of Stochastic Class-based Hard Example Mining for Deep Metric Learning in Tensorflow 2.1.0.  MobilenetV2 is used instead of InceptionV1.


Guide:

 1) Load CUB dataset into root directory CUB_200_2011. Images should be located in CUB_200_2011/images/*
 2) Run Create_CUB_Dataset.py to reshape the CUB images and save them all as .npy files
 3) Run Train_Model.py with parameters to start training

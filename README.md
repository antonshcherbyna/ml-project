# ml-project

### Outline

In this project we want to explore and compare different unsupervised technics for feature extraction. 

The project was completed during ML course at UCU master's program in Data Science.

### How to use this repo 

This repo has the following structure:
 1. dim - implementation of the Deep InfoMax (DIM) algorithm described in the paper [Learning deep representations by mutual information estimation and maximization](https://arxiv.org/abs/1808.06670)
 2. TODO: Vadym, write here about your paper
 3. vectors - here extracted vectors are stored, so you don't need to train model yourself
 4. notebooks - useful notebooks for exploring properties of extracted represantations
 
 If you want to train DIM by yourself you can easily do it by using scripts loacted at `dim/`:
  * for train use ```python train.py --num_epochs 100 --train_batch_size 32 --test_batch_size 16 --lr 0.001 --logdir /where/to/store/logs --chkpdir /where/to/store/weights```  
  If you have more than one gpu, add `--multi_gpu`. Also, you can monitor logs via tensorboard.
  * for vectors extraction use ```python infer.py --chkpdir /path/to/weights/folder --chkpname epoch-100.chkp --outdir /where/to/store/vectors --batch_size 32```  
  If you have more than one gpu, add `--multi_gpu`. It's better to use default parameter for ```--outdir```.
  
  If you want to train *VADYM'S ALGO* by yourself you can easily do it by using scripts loacted at `vadyms' algo/`:
  

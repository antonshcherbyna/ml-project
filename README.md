# ucu-ml-project

## Outline

In this project we want to explore and compare different unsupervised technics for feature extraction. 

The project was completed during ML course at UCU master's program in Data Science.

## How to use this repo 

This repo has the following structure:
 1. **dim** - implementation of the Deep InfoMax (DIM) algorithm described in the paper [Learning deep representations by mutual information estimation and maximization](https://arxiv.org/abs/1808.06670)
 2. **aet** - implementation of the Auto-Encoding Transformations (AET) algorithm described in the paper [AET vs. AED: Unsupervised Representation Learning by Auto-Encoding Transformations rather than Data](https://arxiv.org/pdf/1901.04596.pdf)
 3. **vectors** - here extracted vectors are stored, so you don't need to train models yourself
 4. **notebooks** - useful jupyter notebooks for exploring properties of the extracted represantations
 
 If you want to train **DIM** by yourself you can easily do it by using scripts loacted at `dim/`:
  * for train use ```python train.py --num_epochs 100 --train_batch_size 32 --test_batch_size 16 --lr 0.001 --logdir /where/to/store/logs --chkpdir /where/to/store/weights```  
  If you have more than one gpu, add `--multi_gpu`. Also, you can monitor logs via tensorboard.
  * for vectors extraction use ```python infer.py --chkpdir /path/to/weights/folder --chkpname epoch-100.chkp --outdir /where/to/store/vectors --batch_size 32```  
  If you have more than one gpu, add `--multi_gpu`. It's better to use default parameter for ```--outdir```.
  
  If you want to train **AET** by yourself you can easily do it by using jupyter notebook located at `aet/`: `aet.ipynb`
  
  P.S. Some parts of code were taken from [DIM](https://github.com/rdevon/DIM) and from [AET](https://github.com/maple-research-lab/AET) respectively.
  
## Update
I conducted several experiments with another datasets to check the quality of representations quantitatively. This time alongised with CIFAR-10 I used CIFAR-100 and STL-10. It's well known that CIFAR-10 is a toy dataset for quick experimenting, but results obtained on this dataset aren't representative enough due to small amount of classes. As we need to get some meaningful information to compare unsupervised and supervised methods I decided to take bigger (but yet suitable to iterate quickly) CIFAR-100 and STL-10. Also, I changed configurations of encoder, now I produce only 64 gloabl units as suggested in paper. It forces the encoder to compress data more efficiently (MI estimation network stays the same).  

You can find vectors obtained from new encoder and on new data at ```vectors/```.

Here are results for linear classifier on top of the global feature:  

 | Dataset | Accuracy on top of the features (linear classifier) | Supervised accuracy (from paper) |
 | :------- | :----------------------------------------: | :-------------------------------: |
 | CIFAR-10 | 62.39% | 75.39% |
 | CIFAR-100 | 36.31% | 42.27% |
 | STL-10 | 55.26% | 68.70% |
 
## Future work
1. We want to explore more algorithms in the near future (maybe, during project during our DL course) to compare them. For example, method for feature extraction by [predicting rotations](https://arxiv.org/abs/1803.07728) or [solving Jigsaw puzzles](https://arxiv.org/abs/1603.09246).
2. Performance comparison of the propsed methods by manifold visualization or similarity search are both intuitive and insightful, but they don't provide quantative measure. Such comparison is the subject of the ongoing research itself: how to estimate quality of the representaion overall? Or maybe we want our features to satisfy some specific properties, what's in this case? So we want to put more effort into exploring this question. But for a start, we we'll simply add comparison by linear classifier performance trained on top of the unsupervised features.
3. We see that both approaches that we have explored share similar problem - they don't take into account semantic information. We think that this property is crucial for usefulness of the representations, so we want to do more research in this direction too. 

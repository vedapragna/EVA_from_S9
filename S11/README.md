### Experiment S11: 

    To Train custom ResNet architechture on Cifar10 Dataset using Albumentation Augmentations of choice
    (cutout for sure) and implement one cycle policy to tune the Learning Rate while training the network

![](https://raw.githubusercontent.com/vedapragna/EVA_from_S9/master/S11/New%20ResNet%20Arch.PNG) 
    
 
### Target:
    * Achieve atleast 90% accuracy on Test Dataset of Cifar10 in 24 epochs
    * Proper modularisation of the code 
    * Find best LR to train your model using LR Range test.
        - set the best one as LRmax and LRmax/10 as LRmin
    * Implement One cycle Policy to tune LR while training by increasing LR from LRmin to LRmax from 1st 
      to 5th epoch and anneal the LR from LR max to LRmin from 5th epoch to 24th epoch
    
### Result:
    Trained New Resnet model on Cifar10 dataset starting with:
      * Albumnetation Augmentaions of FlipLR, Pad and cutout, Normalise on Train dataset and 
        just Normalise on Test dataset
      * Learning Rate - Max: 0.046 ; Learning Rate - Min: 0.0046
      * Total Epochs : 24
      * L2 Normalisation is implemented with weight decay = 0.0018
      * Final Training Accuracy = 95% Test Accuracy = 90%
      
     LR Range Test - Traning Accuracy Vs Learning Rate  :
     
![](https://raw.githubusercontent.com/vedapragna/EVA_from_S9/master/S11/LR_Range_test_graph.png) 

    
     Triangular schedule(cyclic learning rate not One cycle policy):

![](https://raw.githubusercontent.com/vedapragna/EVA_from_S9/master/S11/Zigzag.png)


# EVA_from_S9_to_S11

### Experiment S9: 
    To Train ResNet18 Model on Cifar10 Dataset using Albumentation Augmentations and plot the class activation
    map of the Model inferenced on sample Cifar10 image using GradCAM algorithm
    
### Target:
    * Achieve atleast 87% accuracy on Test Dataset of Cifar10 using ResNet18 model
    * Proper modularisation of the code 
    * Implement GradCAM algorithm in a separate Module and plot the Heatmaps the inference using GradCAM 
    module implemented
    
### Result:
    Trained Resnet18 model on Cifar10 dataset starting with:
      * Albumnetation Augmentaions of HorizontalFlip, Normalise on Train dataset and just Normalise on Test dataset
      * Learning Rate: 0.0005 and updated to one third of current Learning rate every 10 epochs once
      * Total Epochs : 24
      * L2 Normalisation is implemented with weight decay = 0.2
      * Final Training Accuracy = 90.28% Test Accuracy = 87.51%
      
     GradCam Plots of the Images inferenced using the trained model are as follows:
     
![](https://raw.githubusercontent.com/vedapragna/EVA_from_S9/master/sample_horse_heatmap.PNG) 

![](https://raw.githubusercontent.com/vedapragna/EVA_from_S9/master/sample_truck_heatmap.PNG)


### Experiment S10:
    To Train ResNet18 Model on Cifar10 Dataset using Albumentation Augmentations of choice (cutout for sure) and plot 
    the class activation maps of misclassified Images using GradCAM algorithm
    
### Target:
    * Achieve atleast 88% accuracy on Test Dataset of Cifar10 using ResNet18 model in less than 50 epochs
    * Proper modularisation of the code 
    * Implement GradCAM algorithm in a separate Module and plot the Heatmaps the inference using GradCAM 
    module implemented
    * Find best LR to train your model
    * Use SGD with Momentum as optimizer and reduce LR on Plateau
    
### Result:
    Trained Resnet18 model on Cifar10 dataset starting with:
      * Albumnetation Augmentaions of HorizontalFlip, cutout, Normalise on Train dataset and just Normalise on Test dataset
      * Learning Rate: 0.0525
      * Total Epochs : 30
      * L2 Normalisation is implemented with weight decay = 0.006
      * Final Training Accuracy = 99.9% Test Accuracy = 90.7%
      
    
### Training and Validation Accuracy Curves:

![](https://raw.githubusercontent.com/vedapragna/EVA_from_S9/master/Outputs/Accuracy%20curves.png)


### GradCam Plots of few misclassified Images inferenced using the trained model are as follows:
     
![](https://raw.githubusercontent.com/vedapragna/EVA_from_S9/master/Outputs/Misclassified_Imgs_HeatMaps.png) 


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



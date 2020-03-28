# EVA_from_S9

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
      * Learning Rate: 0.0832 
      * Total Epochs : 25
      * L2 Normalisation is implemented with weight decay = 0.002
      * Final Training Accuracy = 90.28% Test Accuracy = 88.51%
      
     GradCam Plots of few misclassified Images inferenced using the trained model are as follows:
     
![](https://raw.githubusercontent.com/vedapragna/EVA_from_S9/master/sample_horse_heatmap.PNG) 

    
     Training and Validation Accuracy Curves:

![](https://raw.githubusercontent.com/vedapragna/EVA_from_S9/master/sample_truck_heatmap.PNG)

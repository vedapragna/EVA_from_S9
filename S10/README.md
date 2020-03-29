# EVA_from_S9

### Experiment : 
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
     
![](https://raw.githubusercontent.com/vedapragna/EVA_from_S9/master/Outputs/sample_horse_heatmap.png) 


![](https://raw.githubusercontent.com/vedapragna/EVA_from_S9/master/Outputs/sample_truck_heatmap.png)
    
    

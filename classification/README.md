## Abstract
Wikipedia is full of articles... and images! Having over 53 million articles in 299 languages containing 11.5 million unique images, there is a great need for automated organization of all this data. Inspired by ORES, an ensemble of machine learning systems in Wikipedia that provides among others automated labeling of articles, this project aims at automated \textit{topic} labeling of images in Wikipedia. In this report, experiments are made using images labeled with the ORES labels of the articles where they are present, and with the custom labels that were generated with a heuristic in the taxonomy part of this semester project. Two different models (EfficientNetB0 and EfficientNetB2) are trained on this data using 10 or 20 labels. As the main insights we understood that: 
- the custom labels were inferior to ORES labels according to our metrics; 
- the network with more parameters, EfficientNetB2, yielded higher prediction values having greater average recall but does not outperform EfficientNetB0 with regards to the ROC curves; 
- the labels with better performance are those that are most present in the dataset used in pre-training. 

## Code organization
- [Summary.ipynb](https://github.com/epfl-dlab/wiki_image_classification/blob/main/classification/Summary.ipynb): creates a [dictionary]([url](https://github.com/epfl-dlab/wiki_image_classification/blob/main/classification/training_configurations.json)) with the different training setups that we ran.
- [CleanAndSplitData.py](https://github.com/epfl-dlab/wiki_image_classification/blob/main/classification/CleanAndSplitData.py): filters data from invalid files and splits it into train and test.
- [TrainClassification.py](https://github.com/epfl-dlab/wiki_image_classification/blob/main/classification/TrainClassification.py): loads the training data and performs the fine-tuning using the already split data. 
- [Evaluate.py](https://github.com/epfl-dlab/wiki_image_classification/blob/main/classification/Evaluate.py): loads the test data and the latest training checkpoint and evaluates the model on the test data.

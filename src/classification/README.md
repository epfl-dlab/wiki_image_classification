## Abstract
With over 53 million articles and 11 million images, Wikipedia is the greatest encyclopedia in history. The number of users is equally significant, with daily views surpassing 1 billion. Such an enormous system needs automation of tasks to make it possible for the volunteers to maintain. When it comes to textual data, there is a system based on machine learning called **ORES** providing automation to tasks such as article quality estimation and article topic routing. A **visual counterpart system** also needs to be developed to support tasks such as vandalism detection in images and for a better understanding of the visual data of Wikipedia. Researchers from the Wikimedia Foundation identified a hindrance to implementing the visual counterpart of ORES: the images of Wikipedia lack topical metadata. Thus, this work aims at developing a deep learning model that classifies images into a set of topics, which have been pre-determined in parallel work. State-of-the-art image classification models and other methods to mitigate the existing class imbalance are used. The conducted experiments show, among others, that:
- using the data that considers the hierarchy of labels performs better than the hierarchy-agnostic data; 
- resampling techniques are ineffective at mitigating imbalance due to the high label concurrence; 
- sample-weighting improves metrics; 
- that initializing parameters as pre-trained on ImageNet rather than randomly yields better metrics. 

Moreover, we find interesting **outlier labels** that, despite having fewer samples, obtain better performance metrics, which is believed to be either due to bias from pre-training or simply more signal in the label. The distribution of the visual data predicted by the model is displayed. Finally, some qualitative examples of the model predictions to some images are presented, **proving the ability of the model to find correct labels that are missing in the ground truth**.


## Results



## Documents
This work is the result of a master's thesis. Here is the report and the presentation slides:
- [Report](https://github.com/epfl-dlab/wiki_image_classification/blob/main/reports/matvi959_final_master_thesis_200123.pdf)
- [Presentation](https://github.com/epfl-dlab/wiki_image_classification/blob/main/reports/MasterThesisPresentation_Bernat.pdf)  

## Code organization
- [get_labels.ipynb](https://github.com/epfl-dlab/wiki_image_classification/blob/main/classification/get_labels.ipynb) - Queries the heuristics file to assign labels to the images based on chosen taxonomy and heuristic versions. The resulting file is a dataframe with the url of the images, and their labels as predicted by the heuristics.
- [clean_and_split_data.py](https://github.com/epfl-dlab/wiki_image_classification/blob/main/classification/clean_and_split_data.py) - Filters the dataframe outputed by [get_labels.ipynb](https://github.com/epfl-dlab/wiki_image_classification/blob/main/classification/get_labels.ipynb) from invalid files and splits it into train, validation, and test sets.
- [create_training_config.ipynb](https://github.com/epfl-dlab/wiki_image_classification/blob/main/classification/create_training_config.ipynb) - Creates a [dictionary](https://github.com/epfl-dlab/wiki_image_classification/blob/main/src/classification/training_configurations.json) with different training setups.
- [train_classification.py](https://github.com/epfl-dlab/wiki_image_classification/blob/main/classification/train_classification.py) - Loads training hyper-parameters, loads training and validation data, creates and train model, and plots training metrics. 
- [evaluate.ipynb](https://github.com/epfl-dlab/wiki_image_classification/blob/main/classification/evaluate.py) - Loads the test data, creates the model, loads the best model weights, evaluates the model on the test data and gives out performance metrics.
- [help_functions.py](https://github.com/epfl-dlab/wiki_image_classification/blob/main/classification/help_functions.py) - Help functions used in [train_classification.py](https://github.com/epfl-dlab/wiki_image_classification/blob/main/classification/train_classification.py) and [evaluate.ipynb](https://github.com/epfl-dlab/wiki_image_classification/blob/main/classification/evaluate.py).
[hierarchical_model.py](https://github.com/epfl-dlab/wiki_image_classification/blob/main/src/classification/hierarchical_model.py) - Tensorflow implementation of the [C-HMCNN(h)](https://proceedings.neurips.cc/paper/2020/hash/6dd4e10e3296fa63738371ec0d5df818-Abstract.html) hierarchical multilabel classification model. The adjancecy matrix is hardcoded from taxonomy version 1.2.


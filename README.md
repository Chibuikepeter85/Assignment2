Assigment 2: Development of garbage classification model.
Group 3:  Chibuike Peter Ohanu and Irtaza Sohail 
In this Assignment 2, we developed a garbage classification model that combines image and text data using PyTorch. The garbage dataset is organized into train, validation, and test. This is to ensure consistent image dimensions through padding and resizing.
We applied image augmentations and normalization to enhance model generalization and compatibility with a pre-trained model.
The DistilBERT is used to tokenize text and standardizing length through padding or truncation.
The customdataset creates a dataset class to load and transform image-text pairs for model input.
The model architecture is a hybrid model used in combining ResNet50 for image features and DistilBERT for text, with a final classifier for prediction.
We used weighted cross-entropy loss for class balance with adaptive learning rate scheduling for loss and optimization. 
The model is trained while monitoring validation metrics and saved the best model based on validation loss. 
The results show the evaluated model performance and generates accuracy. The confusion matrix is used to visualize the class-level metrics.

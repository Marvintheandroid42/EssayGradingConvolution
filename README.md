# EssayGradingConvolution

[Project Hosted on Kaggle](https://www.kaggle.com/code/parkersquare/using-convolution-neural-networks-to-grade-essays)

> Using image processing techniques to grade short answers using neural networks

Most methods in processing natural language in the field of machine learning usually revolve around recurrent methods like RNN's or LSTM's which are neural network architectures that allow for models to understand the contextual relationships between a series of input like text. However, in this project, an alternate way of processing natural language is proposed. This method comes from image processing. In image processing, the idea of convolutions is vital to building models. Images can be broken up into various levels of abstraction. A large image can have edges, groups of edges that form shapes, and groups of shapes that can form figures. In order to extract these various layers of abstractions, convolutions can be used. They are a way of downsampling and image by calculating a correlation between a region of pixels and a filter. By checking the similarity between the images and the filter, different aspects of the image like edges can be extracted. In the case of machine learning, these filters can be represented as weights and trained as parameters of a neural network in order to extract salient features of an image that can be used to classify the image or perform regression.

This project uses this same idea of abstraction but in natural language. Language can also be broken into different levels of abstraction. Characters, words, phrases, sentences, and clauses are all substructures of language. Thus this same feature extraction which was applied to a 2 dimensional image can also be applied to a 1 dimensional series. Instead of extracting visual features, here we are comparing a filter which can be thought of as a group of characters or words with the input in order to measure the similarity. This allows us to understand whether certain kinds of structures or collections exist within the input. These measurements of similarity can then be used to perform classification or regression. This idea can be used in grading text. Grading is a regression problem where we are given text as inputs and must output a continuous variable which is the grade. By seeing whether the input has certain features that are learned from looking at the best responses in a dataset, we can produce a feature map of the correlation between the input and these features using convolutions. These feature maps can then be used as input into a neural network in order to output a grade. 

A smaller example of the problem is described below:

>I LOVE TO CODE

This is our string of text that represents the ground truth which gets the highest grade. Now let's say we are given the following input 

> I OVEL OT DCOE

This example is misspelled and thus must receive a lower grade. We can measure if it's misspelled by measuring the similarity to the substrings like "CODE" or "LOVE". If there are similar substrings in the example input then it gets graded higher, if these substrings are omitted then a lower grade is given. In the example above, the substrings occur but the characters are in the wrong order thus it will not get too bad of a grade but neither will it perform well. In order for the computer to be able to calculate a degree of similarity or convolution, we must first transform this text into numbers that the machine can understand. This can be done by encoding. We take every unique letter and assign it a number. This number can then be used to build a sequence of numbers that represent the text. For the example above this could look like

> 1 2345 26 7824

The ground truth on the other hand will look like this:

> 1 5234 62 8274

Now we can measure the degree of similarity between the sequences by choosing a filter and seeing whether it exists in the input. Like before let's take the word "CODE" which is represented as "5234". In the misspelled example, we can spot a similar substring "2345" which is in the wrong order but has the same numbers the degree of similarity is quite high. We can check the degree of similarity for all the other substrings in the example that are 4 characters long. Doing this for more filters, we can see whether the correct words and patters that are observed in the high-scoring examples exist in the given example. This degree of similarity can be quantified mathematically as a number using the convolution function which can be then fed into a neural network to perform regression to predict the grade. In the project, a charecter level encoding was chosen to be able to tell if words are misspelled which is an important aspect when grading text. For other use cases, the encoding can also be done a word level or phrase level basis. We can also encode these words as a larger dimensional vector in order to capture spatial similarity between words and characters as done in the code. 

The training dataset was made up of these numerically converted sentences from text and the grade that they received. The following describes the strcutre of the nerual network:

_________________________________________________________________
Neural Network Architecture 
=================================================================
 input_2 (InputLayer)        [(None, 3000)]            0         
                                                                 
 embedding_1 (Embedding)     (None, 3000, 32)          5312      
                                                                 
 conv1d_1 (Conv1D)           (None, 3000, 32)          3104      
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 1500, 32)         0         
 1D)                                                             
                                                                 
 flatten_1 (Flatten)         (None, 48000)             0         
                                                                 
 dense_3 (Dense)             (None, 16)                768016    
                                                                 
 dropout_2 (Dropout)         (None, 16)                0         
                                                                 
 dense_4 (Dense)             (None, 8)                 136       
                                                                 
 dropout_3 (Dropout)         (None, 8)                 0         
                                                                 
 dense_5 (Dense)             (None, 1)                 9         
                                                                 
Total params: 776,577
Trainable params: 776,577
Non-trainable params: 0
_________________________________________________________________

First, an embedding layer was used to generate the embeddings for the characters into vectors. Then the convolution or similarity calculation was performed. This was done with 32 different filters. This was a key design decision as too many filters could cause the model to be overcomplex for the limited dataset and thus overfit. Too few filters and we don't have enough features to understand the structure of high-scoring text. After the feature map or similarities between the filters are calculated, these values are then passed into a fully connected neural network to predict the grade. These filters that the convolution was calculated from were some of the weights of the neural network. These filters or the high-scoring elements that they represent are learned from the dataset. When training the network, a lot of hyperparameters needed to be tweaked like the number of convolutional layers and size of the filters whether they would be 5 characters long or more. The smaller the filter the smaller the context of finding similarity is for the model. Over the course of training, a validation loss or testing loss of 1.4 was achieved which is very low. The loss function used was mean squared error where numbers closer to 0 are have the least loss or error. Thus the neural network was trained adequatly. Other technques liek ensembling or using LSTM and convolutions together could be used to generate better results.  






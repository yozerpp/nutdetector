### Introduction
This is a program designed to classify small sized objects like nuts and seeds. It's written with C++ and Cuda.
### Pipeline Summary
  ## Shared Steps
- Program performs _binary conversion with K-mean clustering to a Gaussian filter applied grayscale image followed by dilation and erosion to reduce noise. All these algorithms are executed in GPU.
- A single threaded algorithm performs position detection and also filters objects under a threshold calculated from the size of largest object found.
- Every detected object is flattened and fed into shape detection algorithm(that is much like a formalized and simplified four layered neural network)
  ## Training
- Extracted features are first averaged with the objects that share the same label and _detect features are acquired, then min max normalization is performed across labels and result is saved.
  ## Testing
- Every object is labeled by their closest object by the euclidian distance.
### Accuracy
Output and input images are located in 'test' and 'out' folders' you can compare the results yourself.
Currently model seperates very distinct objects like almonds and chickpeas in high accuracy, but fails to distinguish shapes that are in between (like spiced nuts). T think that is because lack of a sophisticated normalization stage and 
T'll look into trying new normalization methods and improving the accuracy.

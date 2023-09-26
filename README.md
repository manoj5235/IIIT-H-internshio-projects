# IIIT-Internship

## IMAGE CLASSIFICATION

This is the implementation of a convolution neural network(CNN) using pytorch to classify images from the CIFAR-10 dataset.

The CIFAR-10 dataset contains 60,000 32*32 color images of 10 classes,with 
6000 images per class.

Here The CNN is to classify these images into one of the ten classes : plane, car, bird, cat, deer, dog, frog, horse, ship, and truck

The implemtation of code starts with

1.Importing Required libraries:

•	I have imported some modules for my image classification they are
             
              1.torch :- it is used for building and training neural networks
              2.torch.nn 
              3.torch.nn.functionoal
              4.torchvision
              5.torchvision.transforms
              6.matplotlib.pypplot
              7.numpy

2.Device configuration:

•	If CUDA is available it uses the GPU otherwise it uses CPU for computation

3.Hyper parameters

•	I have used 3 parameters they are 'num_epochs', 'batch_size', 
'learning_rate'.

4.Data preprocessing.

•	transform is used to converts the PIL images in the dataset to tensors and normalizes the pixel values to the range[-1,1].

5.Loading the CIFAR-10 Dataset.

•	The CIFAR-10 dataset is loaded by using 'torchvision.datasets.CIFAR-10'.

6.Data loaders.

•	and now the dataloaders are created by using 

'torch.utils.data.Dataloader' , which enables efficient loading of data during training and testing.

7.Visualising the images.

•	The 'imshow' function is defined to visualize a batch of images from the dataset

8.Model initialization.

•	An instance of the ConvNet class is created and moved to the device.

9.Loss and optimizer.

•	Here the CrossEntropyLoss is used as the loss function and Stochastic Gradient Descent(SGD) is used as the optimizer for the training the model.

10.Training loop.

•	The model is trrained for the specified number of epochs.For each epoch the code iterates through the batches in the training data and performs forward and backward passes and updates the model parameters using Gradient descent.

11.Testing the model

•	After training, the model is evaluated on the test dataset. The accuracy of teh model on the entire test dataset and the accuracy for each injdividual class are calculated and the results are printed in the console


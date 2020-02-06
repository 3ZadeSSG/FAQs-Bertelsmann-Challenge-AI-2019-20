**Q1: What is the use of Transform in dataloader?**

Resource:
[torchvision.transforms](https://pytorch.org/docs/stable/torchvision/transforms.html)

**Q2: Why I'm getting this error: AttributeError: module 'helper' has no attribute 'view_classify'**

You need to get helper.py file in your repository.You can get this file using following command:

       `!wget -c https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/master/intro-to-pytorch/helper.py`

**Q3: What is the difference between: Output = model(images) and Output = model.forward(images)**

The difference is that all the hooks are dispatched in the call function, so if you call .forward and have hooks in your model, the hooks won’t have any effect. Check this discussion: [Any difference between model(input) and model.forward(input)](https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690)

**Q4: I need help regarding transfer learning using PyTorch**

Resource:
[Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

**Q5: We are taking the log_softmax in forward function, so why do you compute torch.exp(model(images))**

We need probabilities to pick the most probable class, for that we apply Softmax. However, because of the [computational reasons](https://docs.python.org/3/tutorial/floatingpoint.html) when dealing with probabilities, we want to deal with their Log instead. So, instead of outputting Softmax, we output LogSoftmax. The output is no longer in a probability space, but since Log is a monotonic function, we do not really care - still just picking the biggest number. Outputting LogSoftmax also works seamlessly when using NLLLoss: since NLLLoss expects scores per classes, rather than their probabilities, we wouldn’t be able to use the output of Softmax (which outputs probabilities) directly, but fortunately, we already got scores in the form of LogSoftmax! Now, to get the probabilities of the classes rather than their scores, we need to undo the Log operation on Softmax — so, we need to apply Exp.

**Q6: Could anyone tell me what is Top-1 error and Top-5 error of pre-trained networks in torchvision.models?**

First, you make a prediction using the CNN and obtain the predicted class multinomial distribution (∑pclass=1). Now, in the case of top-1 score, you check if the top class (the one having the highest probability) is the same as the target label. In the case of top-5 score, you check if the target label is one of your top 5 predictions (the 5 ones with the highest probabilities). In both cases, the top score is computed as the times a predicted label matched the target label, divided by the number of data-points evaluated. Finally, when 5-CNNs are used, you first average their predictions and follow the same procedure for calculating the top-1 and top-5 scores. Suppose your classifier gives you a probability for each class. Lets say we had only "cat", "dog", "house", "mouse" as classes (in this order). Then the classifier gives something like 0.1; 0.2; 0.0; 0.7 as a result. The Top-1 class is "mouse". The top-2 classes are {mouse, dog}. If the correct class was "dog", it would be counted as "correct" for the Top-2 accuracy, but as wrong for the Top-1 accuracy. Hence, in a classification problem with k possible classes, every classifier has 100% top-k accuracy. The "normal" accuracy is top-1.

**Q7: What is the difference between model.fc1.weight and model.fc1.weight.data?**

Both are actual weight. the difference is the data type, fc1.weight is an instance of Parameter where fc1.weight.data is a Tensor. Parameter is subtype/subclass of Tensor.

**Q8: Why have we used 256 hidden units?**

There is no mathematical logic behind this number; it is just convention. You can pick your own number, or treat it as a hyperparameter and find out what is an optimal number of hidden units.
Resource:
- [How to choose the number of hidden layers and nodes in a feedforward neural network?](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw)


On the other hand nvidia has some guidelines for sizing various parameters of different NN architectures https://docs.nvidia.com/deeplearning/sdk/dl-performance-guide/index.html#choose-params when using their sdk, and they say for example
GPUs perform operations efficiently by dividing the work between many parallel processes. Consequently, using parameters that make it easier to break up the operation evenly will lead to the best efficiency. This means choosing parameters (including batch size, input size, output size, and channel counts) to be divisible by larger powers of two, at least 64, and up to 256.
or
Choose batch size and the number of inputs and outputs to be divisible by 8 (FP16) / 16 (INT8) to enable Tensor Cores
While this does not answer out question about the number of neurons in the hidden layer, it shows that choosing sizes or various dimensions may influence the performance understood by "how long do I need to wait for my result".

**Q9: How do we determine the number of hidden layers?**

Resource:
- [Multi-layer perceptron (MLP) architecture: criteria for choosing number of hidden layers and size of the hidden layer?](https://stackoverflow.com/questions/10565868/multi-layer-perceptron-mlp-architecture-criteria-for-choosing-number-of-hidde)

**Q10: Is there any difference between using nn.CrossEntropyLoss and the combination of nn.LogSoftmax() and nn.NLLLoss()?**

As per the [PyTorch docs](https://pytorch.org/docs/stable/nn.html?highlight=crossentropy#torch.nn.CrossEntropyLoss) , there is no difference. You could also refer to this [discussion](https://discuss.pytorch.org/t/what-is-the-difference-between-using-the-cross-entropy-loss-and-using-log-softmax-followed-by-nll-loss/14825) on the Pytorch forum.

**Q11: What are PyTorch tensors?**

As per [Pytorch docs](https://pytorch.org/docs/stable/tensors.html#torch-tensor),
A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.
If you want to know more about tensors, you could also refer to slides or handouts of 1.4, 1.5 and 1.6 of this [Deep Learning course at EPFL](https://fleuret.org/ee559/).

**Q12: When I tried to run the model on GPU in my local machine it gave CUDA Driver Error/RuntimeError**

Most ML Libraries require Nvidia GPU with Compute Capability 3.0 or higher. Same is for PyTorch. GoTo following link and take a look at your GPU Model Number under category, if the compute capability is less than 3.0 then you can't run on GPU. Link : https://developer.nvidia.com/cuda-gpus But if you want you can build PyTorch from source if you really want to run on an old GPU: https://discuss.pytorch.org/t/pytorch-version-for-cuda-compute-capability-3-0-gtx-780m/15889 . But this will not give you a high speedup due to less CUDA Cores and memory bandwidth of those old GPUs

**Q13: Hello, can someone explain to me what is "batch" size?**

it is the number of samples that going to be propagated through the network. For instance, let’s say you have 1000 training samples and you want to set up batch_size equal to 100. Algorithm takes first 100 samples (from 1st to 100th) from the training dataset and trains network. Next it takes second 100 samples (from 101st to 200th) and train network again. We can keep doing this procedure until we will propagate through the networks all samples,

**Q14: After definig my classifier for transfer learning i got error when i ran the training loop 'loss.backward()' - (RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn)**

First make sure you attached the classifer after freezing the parameters of pretrained model, so that your classifier didn't got freezed as well. Second your optimizer should be something like this

     optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)

Third problem might be that the model you are using doesn't had any classifier layer, for example resnet model have the linear layer named 'fc' in that case you need to first take a look at the layer name, then attach the classifier to that layer, so for example in case of 'fc' you should attach the classifier and define optimizer as:

     model.fc=classifier
     optimizer = optim.SGD(model.fc.parameters(), lr=0.01)
 
**Q15:transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) Why do we need it, and what it do?**

Transforms are image transformations and helps us preprocess our data. They can be chained together using Compose().
Using the value 0.5 will transform it in the range of (-1,1) which helps the model learn faster, If you don't normalize your inputs between (0,1) or (-1,1) you could not equally distribute importance of each input, thus naturally large values become dominant according to less values during neural network training,
transforms.Normalize(mean, std)
mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), we have 3 values since its RGB
since we used transforms.ToTensor() it will convert our input data from 0-255 to 0-1 range
we normalize it using the formula: input = (input - mean) / std
minimum value for input which is 0 will be converted to (0-0.5)/0.5=-1
maximum value for input which is 1 will be converted to (1-0.5)/0.5=1


**Q16: In 5.1 it reads at the very end"our nanodegree programs also provide gpu workspaces in the classroom, as well as credits for AWS" does it apply to our scholarship as well?**

AWS credits will be provided for phase-2 (i.e Full Nanodegree program) but for this phase, you can either use Google Colab or Kaggle if you want to access the GPU resources (Both Free).

**Q17: hello everyone hope you are doing great can someone please help me with implementing softmax**

The formula can be interpreted as normalizing every element in a given row by dividing by the sum of the elements in this row, but before the exponent function is applied.
With numpy you could start experimenting like this:

              >>> import numpy as np
              >>> a = np.array([[1,2],[2,3],[3,4]])
              >>> a.shape
              (3, 2)
              >>> a
              array([[1, 2],
                     [2, 3],
                     [3, 4]])
              >>> np.sum(a, axis=1)
              array([3, 5, 7])
              >>> a / np.sum(a, axis=1)
              Traceback (most recent call last):
                File "<stdin>", line 1, in <module>
              ValueError: operands could not be broadcast together with shapes (3,2) (3,)
              >>> np.sum(a, axis=1).reshape(-1, 1)
              array([[3],
                     [5],
                     [7]])
              >>> np.sum(a, axis=1).reshape(-1, 1).shape
              (3, 1)
              >>> a / np.sum(a, axis=1).reshape(-1, 1)
              array([[ 0.33333333,  0.66666667],
                     [ 0.4       ,  0.6       ],
                     [ 0.42857143,  0.57142857]])

Now instead of just dividing the elements of rows by the sum in the corresponding rows, use the exp function as in the formula. The remaining part is how to translate numpy into pytorch.

**Q17: Hi everyone, I have a few questions regarding the code provided in notebook.
What does this command mean? Only one batch is loaded or many batches?**

images, labels = next(iter(trainloader))
How does this command work? running_loss was initialized at 0, are we adding values to it? what does item() mean?
running_loss += loss.item()
Thanks a lot for your help.

Let's start with the first one: images, labels= next(iter(trainloader))
The trainloader is an iterable object in python and the way we take a single batch from that iterator is by calling next(iter(iterable)). A single may contain multiple examples, the batch size is defined when we declare the train loader.
loss.item() is necessary because although the loss is a scalar value, pytorch still considers it a tensor and you cannot add a primitive number type to a tensor object so the item takes the value of that scalar tensor.
The len(trainloader) return the number of inner iterations of the epochs. This is equivalent to the number of samples of that dataset (for MNIST training set it's 50,000) divided by the number of batches. If we set the batch size to 64, we have len(trainloader) = ceil(50000/64) = 782

Q18: I'm doing a review of 5.14 and I'm not quite sure what does thi aterisk do? equals = top_classes == labels.view(*top_class.shape)
In Python, the asterisk is used as an unpacking operator in function
calls. This means it can be used to unpack an iterable like a tuple into the arguments in the function call. You can read more about it here: https://treyhunner.com/2018/10/asterisks-in-python-what-they-are-and-how-to-use-them/
In your case the top_class.shape is a tuple of array dimensions containing a
value for the number of rows and the number of columns of the 2D tensor top_class . This tuple is then unpacked to pass both contained values individually to the function as input parameters.

**Q19: In notebook 5.14 I can't get is the part of the accuracy. I get that the accuracy sort of calculates the mean of the correct responses per batch. But how are we able to use that to calculate the overall accuracy (or mean). I don't know if finding the mean of a mean gives a total mean.**

The last batch, if its size differs from the previous batches, will cause some calculation errors with the accuracy unfortunately, but you still get a pretty close number to the real accuracy
About the mean of the mean, compare the small example

              >>> import numpy as np
              >>> l1=np.array([1,1,0])
              >>> l2=np.array([1,0,0])
              >>> (l1.mean()+l2.mean())/2
              0.5
              >>> (np.sum(l1)+np.sum(l2))/6
              0.5

with

              >>> import numpy as np
              >>> l1=np.array([1,1,0])
              >>> l2=np.array([1,0])
              >>> (l1.mean()+l2.mean())/2
              0.58333333333333326
              >>> (np.sum(l1)+np.sum(l2))/5
              0.59999999999999998

**Q20: Can anyone help me with this. In lesson 5.20 transfer learning, why they have used OrderedDict() in nn.sequential(). Can't we do by adding the layers and activation functions directly to it, as we have done previously?**

You can, OrderedDict helps you name your layers to retrieve them in a more human readable way.
For example, if you want to access the weights, instead of using mode[0].weights
You call model.fc1 (fc1 : fully connected layer 1 is way more readable and cleaner than 0, 1 .. 2
It is also important to use OrderedDict and not just a dict to have the layers added in order as they are defined https://pymotw.com/3/collections/ordereddict.html


**Q21: Hi All, i have a question regarding the validation step. In two different classes validation step syntax were slightly different. For example in one case validation is handled within
with torch.no_grad():
whereas in another class,  after the below
model.eval()
So what are the differences if there is any?**

1.toch.no_grad() is for turning off the history of the matrix weights (pytorch automatically keeps the weight matrices in memory so it can update them during the optimizer.step( ) i.e. when we're performing backprop.) since during evaluation we're only using forward propagation and do not perform backprop. The reason for doing this is that it reduces computational complexity during model evaluation which helps in saving some precious time
2.model.eval() is used for turning on the nodes that were disabled during dropout since we want to evaluate the result of our whole network i.e. we want evaluate the whole model and not leave out any neurons that were turned off during forward prop.
I hope this helps you understand better.

**Q22: How to turn on CUDA on colab and kaggle?**

In Google Colab, go to Runtime -> Change runtime type -> Hardware accelerator and select GPU from the dropdown list

In Kaggle the settings are on the right side of your screen (in a sidebar) and there should be a button to turn GPU on.


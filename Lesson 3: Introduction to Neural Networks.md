# Lesson 3 : Introduction to Neural Networks

**Q1: I am having some trouble understanding backpropagation when training the neural net.**

Resource:

[Michael Nielsen: Neural Networks and Deep Learning - Chapter 2](http://neuralnetworksanddeeplearning.com/chap2.html)

[Getting Started with PyTorch Part 1: Understanding how Automatic Differentiation works](https://towardsdatascience.com/getting-started-with-pytorch-part-1-understanding-how-automatic-differentiation-works-5008282073ec)

**Q2: What's the perceptron algorithm?**

  Resource:
  
  [What the Hell is Perceptron?](https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53)

**Q3: How to find the optimal learning rate?**

  This paper by Leslie Smith is a great resource in finding the optimal learning rate: [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820).
  
  [You can find implementation of this paper in this blog: Estimating an Optimal Learning Rate For a Deep Neural Network](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0)

**Q4: What is cross entropy Loss?**

  Resource:
  
  - [Understanding binary cross-entropy / log loss: a visual explanation](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)
  
  - [Cross - entropy notes from](https://drive.google.com/file/d/1zCPZ1cnwaqVYzEL54WFcc7J34IZDYSRf/view)

**Q5: What is bias?**

  Resource:
  
  - [Role of Bias in Neural Networks](https://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks)
  
  - [But what is a Neural Network? | Deep learning, chapter 1](https://www.youtube.com/watch?v=aircAruvnKk)

**Q6: What is Gradient Descent?**

  Batch Gradient Descent also just known as Gradient Descent usually loads in the entire training examples (dataset) into the network at one go and update the weights based on all the training examples. Stochastic Gradient Descent loads 1 training example at one go and update the weights using only that training example. Lastly, Mini-Batch Gradient Descent is a combination of the two. Mini-Batch Gradient descent instead of taking the entire dataset takes in N batch size. Where N is the number of training examples you can choose. These N training examples are loaded into the network and are used to update the weights once. And subsequent N batches will continue to update the weights until the entire data has been seen.
  Resource:
  Gradient descent, how neural networks learn | Deep learning, chapter 2
  Gradient descent notes 

**Q7: In softmax function why do we take exponential?**

[In softmax classifier, why use exp function to do normalization?](https://datascience.stackexchange.com/questions/23159/in-softmax-classifier-why-use-exp-function-to-do-normalization)

**Q8: How to calculate derivative of sigmoid function?**

[Calculating derivative of sigmoid function](https://drive.google.com/file/d/18JAQsH285lWkXTlvxaPNYiTdjtUC1NI6/view)

**Q9: Why do we need activation function?**

Hi, the purpose of an activation is to introduce non-linearity into the neural network. Essentially, when we are first building Neural Networks, the formula where, y = w1x1 + w2x2 + b is a linear function, this means that it can only linearly separate data points using a line. Adding the non-linearity i.e. activation function allows the model to form different boundary instead of it just being a line.
Resource:

[Activation functions and it’s types-Which is better?](https://towardsdatascience.com/activation-functions-and-its-types-which-is-better-a9a5310cc8f)

**Q10: What's the difference between np.dot() and np.matmul(),for matrix multiplication and when to use them?**

Resource: 
[numpy.dot vs numpy.matmul](https://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication)

**Q11: Having a problem with gradient descent ?**

Here is a [tutorial](https://github.com/bhargitay/Facebook-Pytorch-Challenge-Notes/blob/master/Gradient%20Descent/Gradient_Descent.ipynb) on Gradient Descent with numpy (using the notebook provided by Udacity). Created by @Beata.

**Q12: Are there any notes for these lessons?**

There are notes created by our fellow scholars. You can refer to these notes through [this spreadsheet](https://docs.google.com/spreadsheets/d/1b7eD6dgWXgFuFpbWHImC5lovWLBfPR_zgaedBRA_21s/edit?usp=sharing) created and maintained by @DylanGoh.

**Q13: For the second example, where the line is described by 3x1+ 4x2 - 10 = 0, if the learning rate was set to 0.1, how many times would you have to apply the perceptron trick to move the line to a position where the blue point, at (1, 1), is correctly classified?**
  
  -you need to apply the perceptron trick to shift it in the positive area, and count how many steps it will take,
  
        1st step:
        3(1) + 4(1) - 10 = 0
        3 + 0.1 = 3.1
        4 + 0.1 = 4.1
        -10 + 0.1 = 9.9
        
  then:
  
        3.1 + 4.1 - 9.9 = -2.7, it's still negative so apply the trick again
        2nd step:
        3.1 + 0.1 = 3.2
        4.1 + 0.1 = 4.2
        -9.9 + 0.1 = 9.8
        3.2 + 4.2 - 9.8 = -2.4, it's still negative so apply the trick again
        
  repeat the trick until it is equal or greater than 0 and count how many steps you did

**Q15: Why is that we can’t use values such as 0, 1 and 2 for classifying the animal as Duck, Walrus or Beaver? Why does this assume dependencies between classes?**
Answered by: @Carlo David
We can't say 0 = Duck, 1 = Walrus, 2 = Beaver, because our model think that a high number is better than a lower number, its like telling the model Beaver is better than Walrus, and Walrus is better than Duck. So we want to avoid that, instead we perform one hot encoding. It tries to avoid the natural ordered relationships, because our model will naturally give the higher numbers with higher weights.

**Q16: When and why is One-Hot-Encoding used?**
-Quick answer: @Nicolas Remerscheid
Used for multi-class-classification (non-linear).
Background: A multic-class-classifier can be viewed as the combination of seperate single classifiers: One-Vs-All
So essentially the error function of a multi-class-classifier can be viewed as the sum of the errors of all the separate classifiers; or: the error of each data-example from each classifier.
What is needed?: The error is always determined by the difference to a correct given value! That means one needs a given true output for each data point for each classifier
What is given?: In the given data-set there is only one correct true value for each data-example/point which is equal to: _ the number of the correct class_
What is the solution?: The one value indicating the correct class by a number from: [1 - #classes] has to be transformed from a scalar to a vector which has one entry for each classifier and either says 0 (not correct class) or 1 (correct class).
--> I.e.: if y1 = 3 and there would be 5 different possible classes it has to be converted to y1 = [0, 0, 1, 0, 0];
- [Additional Ressources](https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159)

**Q17: A little terminology question: In the video the instructor uses the terms error-function and activation-function. Is it the same or is it something different?**

They are different:
- An activation function is used to modify the value that a perceptron/node of NN outputs. So this function is applied in the output of each node in the feed forward phase.
- An error function is used to measure how correct is a prediction compared to the real value of a label. This measure is used to modify the weights in a NN. So this function is used in the Backpropagation phase.

**Q18: in exercise 3.26 - why the display function call parameters uses weight[0]/weight[1] , similarly for bias and both are negative? please help!

Starting from the standard form equation of a line:

        w0 * x0 + w1 * x1 + b = 0
        you arrive at the usual y = a * x + b form, where x1 = y and x2 = x:
        x1 = - w0 / w1 * x0 - b / w1
There is a section about the standard form equation of line in khan academy https://www.khanacademy.org/math/algebra/x2f8bb11595b61c86:forms-of-linear-equations#x2f8bb11595b61c86:standard-form

**Q19: Hi. Can someone please explain how do we multiply the weights when we have a matrix 3x2 and the inputs are 3x1?**

                          W11^2             W11^1 W12^1 x1
                        y_hat = sigma  W21^2 sigma W21^2 W22^1 x2
                          W31^2             W31^2 W32^1 1

This is to be understood in terms of functions of matrices (and compositions of functions). It's not about the matrix operations. In the later lectures examples involving the familiar matrix operations are given.
Apart from that you can take transpose of 3x2 matrix so that it is 2x3 and then you can multiply it with your 3x1 input


**Q20: Is higher the dimensions in features of a model then more will be the accuracy in classification?**
One hot-encoding is obvious for target but for categories in features is done to get accuracy.
In image of Lesson 3.35 there are four ranks under rank category, therefore I think one hot encode is implemented to get more accuracy with more dimensions.

The increased number of features is a by-product of the hot-encoding applied to the original rank feature. Typically, the more dimensions the more problems, due to various factors: https://en.wikipedia.org/wiki/Curse_of_dimensionality, correlation between features https://www.kaggle.com/reisel/how-to-handle-correlated-features, lack of sufficient amount of data for the amount of features, ...

**Q21: Why testing is required in model?**

- 1) training set is the dataset on which you train your model
- 2) validation dataset is the dataset which is used to tune the hyper parameters of our model so that it generalizes better. More like a model validator
- 3) test dataset is more like the simulation of real world data for which we might not know the class label.

In order to tune the performance of the model one of the methods is to further split the training set into the training set proper and a validation set. Then you tune the performance of the model using the validation set, but the performance on the test set must be evaluated only once. See https://machinelearningmastery.com/difference-test-validation-datasets/
There is a funny comment by Yaser Abu-Mostafa https://www.youtube.com/watch?v=EZBUDG12Nr0#t=1h00m47s - there problem with "data snooping" into the test set, appears also when using popular datasets together with architectures designed by "others".

**Q22: Is it OK to post code on github?**

Yes, this course repository https://github.com/udacity/deep-learning-v2-pytorch is released under the MIT license.
On github, the proper way of "copying" another repository onto your gitub account is to make a fork https://help.github.com/en/github/getting-started-with-github/fork-a-repo, and keep working on the fork.

Note however, the project code is covered by Udacity Honor Code https://udacity.zendesk.com/hc/en-us/articles/210667103-What-is-the-Udacity-Honor-Code-: you are not allowed to copy other's solutions

I confirm that this submission is my own work. I have not used code from any other Udacity student's or graduate's submission of the same project. I have correctly attributed all code I have obtained from other sources, such as websites, books, forums, blogs, GitHub repos, etc. I understand that Udacity will check my submission for plagiarism, and that failure to adhere to the Udacity Honor Code may result in the cancellation of my enrollment.

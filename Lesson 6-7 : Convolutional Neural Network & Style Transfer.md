**Q1: Why use a pooling layer?**

To progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting.
Resource:
- [cs231n - Pooling layer](http://cs231n.github.io/convolutional-networks/#pool)

**Q2: When trying to run the conv_visualization in my notebook, I'm getting the error ModuleNotFoundError: No module named 'cv2'**

Possible solutions:

      pip install opencv-python
      conda install -c conda-forge opencv

**Q3: Why am I getting this error: Invalid argument 0: Sizes of tensors must match except in dimension 0. Got 499 and 442 in dimension 2 at... ?**

Your images are not all the same size.
Possible solutions:

      transforms.RandomCrop(224)
      transforms.RandomResizedCrop(224)
      transforms.Resize(224)
      transforms.CenterCrop(224)

**Q4: While using transforms.Normalize, we pass in a list of means and a list of standard deviations (std) to normalize the input color channels. How do we define means and std values and why is it 0.5 in some cases?**

Ideally, you should use the mean and standard deviation for each channel. In the case of Imagenet, you use these precalculated values normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). The reason for using 0.5 in the case of mnist is to reduce complexity for the readers. Check Soumith Chintala’s comment here.

**Q5: In weight=torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor) what does unsqueeze(1) mean?**

One of the reasons why we need to unsqueeze which adds in an additional dimension is due to how PyTorch passes in the CNN parameters.(N,Cin,H,W) is the input format to all CNNs defined in PyTorch, where N is the batch size, Cin is the number of channels i.e. grayscale = 1 and color = 3, H is the Height of the image, and W is the Width of the image. It is a neat trick to introduce a dimension of size 1 (using unsqueeze) and since if we are dealing with grayscale images where the number of input channels of grayscale is 1, we can just unsqueeze index 1 (which is the second position/dimension) of the tensor. With the help of @Mitch Deoudes for clarification on this particular question: In this particular case, if you look at the init() function, the "weight" variable is actually being used to directly set the weights of the conv layer. The weight variable is a (4,1,4,4) matrix defined as [output_channels, input_channels, height_of_filter, width_of_filter] as our input provided to the nn.Conv2d is in the format of [input_channels, output_channels, height_of_filter, width_of_filter]. Remember to multiply 2 matrices we have to ensure that the dimensions of the first column and first row to be matched i.e. (NxM x MxO) etc. Therefore, since we are dealing with grayscale we can use a .unsqueeze(1) to introduce a 1 in the second dimension - yet another neat trick.

**Q6: Hello everyone! In Lesson 6.25, we apply flitered_image = cv2.filter2D(gray, -1, sobel_y) to apply the filter. The comment says -1 is the bit-depth. What does this mean exactly?**

It means the output to have the same depth as the source.
https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#filter2d provide the following signature of the method
cv2.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) → dst
and says the following about the meaning of using -1 as ddepth
when ddepth=-1, the output image will have the same depth as the source


**Q7: What does Content Representation mean?**

When you go to deeper layers of a convolution network then these layers represent high level features, for example at the initial layers you would have edges, however the later layers would make use of these edge to recognize shapes in the images. Hence, the layers deeper in the network usually have more precision at recognizing content rather than the background. That is what you mean by content representation. So now if you take the output of a different image from the same layer then the layer would have tried to also find similar content in the other image, and hence reducing the loss between these would result in the content of the two images becoming similar.

**Q8: What does tensor.to("cpu").clone().detach() mean?**

tensor.detach() creates a tensor that shares storage with tensor that does not require grad. tensor.clone() creates a copy of tensor that imitates the original tensor's requires_grad field. You should use detach() when attempting to remove a tensor from a computation graph, and clone() as a way to copy the tensor while still keeping the copy as a part of the computation graph it came from.

**Q9: Whats the point of style transfer in relation to image classification ?**

Style transfer is another method other than classification. You get a stylistic ai- designed image with style transfer whereas you can predict which class a photo belongs to with classification. Use case for style transfer could be for aesthetics reasons.

**Q10: If you are facing this error AttributeError: module 'PIL.Image' has no attribute 'register_extensions'?**

Make sure to insall pillow, you can use this command 

    !pip3 install Pillow==4.1.1.

**Q11: can anyone tell me what does the BytesIO package is doing in style transfer exercise**

BytesIO is doing essentially helping to read the image from path. However it's much faster than the normal open() method we use. Check this link out to learn why we use BytesIO in more detail: https://stackoverflow.com/questions/42800250/difference-between-open-and-io-bytesio-in-binary-streams
io.BytesIO present in the Style_Transfer_Solution.ipynb notebook reads the http response as "bytes", without trying to decode the characters found or interpret newlines, etc.
If you are unsure what a specific python standard library function does try to search directly in the python docs
https://docs.python.org/3/library/io.html#binary-i-o
or just in google search.




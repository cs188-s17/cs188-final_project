#### CS 188b: Computational Methods for Medical Imaging

# **Final Project: 2D Cerebral Blood Vessel Segmentation**
###           _Sneha Venkatesan, Tanya Lohia_

# ______________________________________________________________________________________________________________________________________
## Usage
In order to run this program, first ensure that the before and after data is in 64x64 format, in folders named "before" and "after" respectively, in your directory with the program files. This program requires both python2 as well as numpy and scipy installed. Images with transparent backgrounds should be formatted likewise so that they are processed by the program. In order to run the program from your current directory, enter the following into the command prompt: 
```
        python2 test_models.py
```

Your output will be a png file showing the expected image, as well as the image predicted by the four models we tested our data on, namely, k-nearest-neighbors, extra trees, linear regression and ridge. 

## Introduction

Strokes are the third leading cause of death in the United States, with over 140,000 people dying every year due to strokes and their complications. In addition, they also lead to brain aneurysms and are the leading cause of serious long term disabiliies. While in the past few years the number of deaths causded by strokes has decreased, they continue to be a challenge for doctors to predict and treat effectively. With increasing population, resources to treat patients who have suffered from a stroke remain insufficient. Fortunately, technological advances, particularly machine learning tools can assist in this process by automating one of the most time consuming tasks: Blood vessel segmentation!

One way to assist prediction for strokes is blood vessel segmentation. It is critical for numerous medical applications including diagnosis of issues like arteriosclerosis, malfrmations in arteries and veins, tears in arterial linings and predicting lesion growth for ischemic stroke patients. Blood vessel annotation is primarily done using photoshop and can be incredibly labour intensive and time consuming, which can cause considerable setbacks to treatment processes. 

Several methods have been developed in an effort to automate this process in order to make it faster and more accessible. Currently, the most popular tools for this purpose are pattern classification and machine learning methods. Machine learning application is becoming increasingly popular for biomedical applications and can serve as useful aids for diagnosis. Our aim with this project is to test several machine learning algorithms to discern which are best in the context of blood vessel segmentation.

## Methods

For the machine learning component we used Python's Scikit-Learn packages, and used a regression tool for segmenting vessels. This was originally a "face-completion" project. This means that it took images of faces and ran four different machine learning algorithms on them. Then printed the original top half of the face and the machine learning algorithm's output as the lower half to compare similarities and effectiveness of the various algorithms. We used this structure and adopted it to our requirements. 

Professor Scalzo providded us with the required data which was in the form of two files of images. An original image data file consisting cerebral angiography scans of people who suffered from an ischemic stroke, and an annotations file which consisted of corresponding annotations illustrated using Photoshop. 

![Image of Data](images/img_data.png)                   
![Image of annotation](images/img_anno.png)
### Figure 1: Example of Original Cerebral Angiography Scan (_left_), Example of Corresponding Hand Drawn Annotation (_Right_)


Our aim was to feed both the images as input in the form of pixel arrays and train the algorithms to produce the annotated images themselves given new cerebral angiography data scans. There are four regression based algorithms employed and they help compare the results to determine which one produces the most accurate images. These are:
1.  Extra Trees (Extremely Randomized Trees)
   i.  Extra trees is a type of decision tree algorithm, it is an _ensemble learning method_ which means that it uses multiple learning        algorithms simultaneously and thus obtains more accurate results than any one algorithm. This particular one is a variation of          the _Random Forest_ algorithm. It constructs several decision trees based on the data and finnaly outputs the most occuring one.        The _Extra Trees_ algorithm further randomizes top-down splitting in the tree learner, this assists in decreasing variability.  
2.  K-nn (K-nearest neighbor)
   i.  This is an example of a non-parametric regression algorithm, which means that it is not defined by a bounded set of parameters          and uses _instance-based learning_ giving it more freedom in the weights it assigns. The K-nn regression model creates a function        _h(x)_ that solves a linear regression problem with **_K_** points around a given point **_Xq_**.   
3.  Linear Regression
   i.  This method borrowed from statistics uses a linear equation such as **y = _B1_ x + _B0_** and tries to optimize the values of            **m** and **b** from a given set of values for **x** and **y** so it can minimize error and eventually predict **y** from a              particular value of **x**. It is a parametric regression model, which means that it uses a fixed number of parameters to predict        relationships.
4.  Ridge Regression
   i.  This is a variation of the linear regression model that is used most often when the predictor variables are strongly correlated.        If no particular **x** produces a satisfying answer in the ordinary least squares linear regression method described above, then        the ridge regression model provides a better fit. It does so by purposefully introducing bias to the value of **_B_** in an              attempt to reduce the variability of the estimate.
   
## Evaluation

The promary method we used to evaluate the data was visual comparison to the expected output. Considering that the data consists of pixels arrays of size 64x64 extracted from cerebral angiography scans of patients at the UCLA Medical Center who suffered from ischemic strokes, we also recognize that they do not provide a wholistic representation of all brain scans for blood vessels. The questions we considered were, if the results would improve with multiple iterations of the experiment. How parametric models would perform as compared to non-parametric, and the implications of the size of the data set amongst others.

## Results 

Ultimately, only two of the four algorithms proved fairly efficient in learning how to predict drawings from the Cerebral Angiography scans. An example image of the results are as follows:

![Image of Results](images/final%20results.png) 
### Figure 2: Example of Output Results

As is evident in *figure 3* the clearest and most accurate results were produced by the Extra Trees algorithm. The Ridge and Linear Regression algorithms produced similar results in that they were successful in predicting the right base structure but the image is significantly obscure. The K-nn model is slightly clearer but there is still a considerable amount of noise in the background with too many extra vessels. In the Extra Trees model, although it is the most accurate, the exact blood vessels are not conspicuous enough to substitute in for the hand drawn annotations.  

## Discussion

Part of the reasons for the aforementioned trends could be in the structure of the algorithms themselves. The Linear Regression and Ridge algorithms are both models based on linear regression models which along with being one of the oldest and most studied machine learning algorithms, are also less reliable in situations where the input data has more noise. It assumes that the relationship between the input and output is linear and so it cannot train well for situations where the linearity of the relationship is less pronounced, resulting in a considerable amount of noise in the resulting output. 

The greatest limiting factor with the K-nn model is the _K_ value chosen, too small a _K_ value can cause overfitting where as too large a value can cause underfitting. But, nevertheless, it is better suited for this application as it provides more accuracy when the data set itself has a considerable amount of background noise.

The Extra Trees model produced the best results because its algorithm employs the most amount of randomness of the four tested here in the predictions it makes for optimization. It uses multiple inputs, makes random predictions on each and then integrates them to form a combined decision tree like algorithm which is most effective for a model in which the input images are not clear themselves. 

Another considerable limiting factor was the amount of training data provided. There were only a total of 86 pairs of scanned and annotated images to use for training the algorithm. Computational power of the computer was also a minor issue since the largest sized picture that could be used was 64x64 which does not offer an optimal amount of clarity in terms of the number of values for pixels in each array.

Further steps for this study would entail obtaining more data to use for training purposes. Other algorithms could also be sampled which may lead to a more accurate reproduction, such as neural networks like KSOM, or the back-propagation algorithm which do not lie on any specific models and are much more malleable, hence adapting better to the training data. Neural Network algorithms can be considered as a _nonlinear regression_ tool, hence providing a more precise learning model for data that is not easily charaterized using one model.




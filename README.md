#### CS 188b: Computational Methods for Medical Imaging

# **Final Project: 2D Cerebral Blood Vessel Segmentation**
###           _Sneha Venkatesan, Tanya Lohia_



## Introduction

Strokes are the third leading cause of death in the United States, with over 140,000 people dying every year due to strokes and their complications. In addition, they also lead to brain aneurysms and are the leading cause of serious long term disabiliies. While in the past few years the number of deaths causded by strokes has decreased, they continue to be a challenge for doctors to predict and treat effectively. With increasing population, resources to treat patients who have suffered from a stroke remain insufficient. Fortunately, technological advances, particularly machine learning tools can assist in this process by automating one of the most time consuming tasks: Blood vessel segmentation!

One way to assist prediction for strokes is blood vessel segmentation. It is critical for numerous medical applications including diagnosis of issues like arteriosclerosis, malfrmations in arteries and veins, tears in arterial linings and predicting lesion growth for ischemic stroke patients. Blood vessel annotation is primarily done using photoshop and can be incredibly labour intensive and time consuming, which can cause considerable setbacks to treatment processes. 

Several methods have been developed in an effort to automate this process in order to make it faster and more accessible. Currently, the most popular tools for this purpose are pattern classification and machine learning methods. Machine learning application is becoming increasingly popular for biomedical applications and can serve as useful aids for diagnosis. Our aim with this project is to test several machine learning algorithms to discern which are best in the context of blood vessel segmentation.

#
## Methods

For the machine learning component we used Python's Scikit-Learn packages, and used a regression tool for segmenting vessels. This was originally a "face-completion" project. This means that it took images of faces and ran four different machine learning algorithms on them. Then printed the original top half of the face and the machine learning algorithm's output as the lower half to compare similarities and effectiveness of the various algorithms. We used this structure and adopted it to our requirements. 

Professor Scalzo providded us with the required data which was in the form of two files of images. An original image data file consisting cerebral angiography scans of people who suffered from an ischemic stroke, and an annotations file which consisted of corresponding annotations illustrated using Photoshop. 

![Image of Data](https://github.com/zwangzob/cs188-final_project/blob/master/img_data.png)
![image of annotation](https://github.com/zwangzob/cs188-final_project/blob/master/img_anno.png)


Our aim was to feed both the images as input, and train the algorithms to produce the annotated images itself given new cerebral angiography data scans. 



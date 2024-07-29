
# Contrastive Learning Pipeline




## Overview

Medical imaging data is abundant, but the overwhelming majority of it is unlabeled. For problems of patient outcome prediction or tumor segmentation, we may have lots of MRIs, CTs, and other scans, but how can we train a good model if only a small subset of it is actually labeled with "tumor" or "no tumor?"

This brings us to unsupervised or self-supervised methods, which allow us to harness the large amounts of imaging data present in healthcare to pretrain deep learning models. In particular, contrastive learning is one particular method that has gained in popularity recently. 

I undertook this project in my junior year summer as part of the Chiang Lab within the Computational Medicine Department at UCLA. This was a pilot project, so the main goal for me was to build a contrastive learning pipeline based on the 2020 paper, [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709). Ideally, this framework would then be applied to different problems the lab studies, such as age-related macular degeneration or severe traumatic brain injury. 

In this README, I will be mostly detailing the intuition behind each part of the framework. Here and there, I will also talk about the code I wrote and how I separated it into modules, but for more information, feel free to check out the code, which I tried to comment/provide documentation for whenever necessary. 

I wrote two versions of this contrastive framework, a custom implementation and an implementation using MONAI, a PyTorch-based framework specialized for deep learning in health. So, files like loss.py for example will differ between the two because in the MONAI implementation, I just call their ContrastiveLoss module, whereas in the custom implementation, I implement the contrastive loss function from scratch.

Lastly, in the main directory you will see two Jupyter Notebooks, which go through an overfit experiment (more on that later) and a general demo of my MONAI-based contrastive learning framework.

## Dataset

I worked with three datasets: MNIST, a retinal optical coherence tomography (retinal OCT) dataset, and a 3-D brain MRI dataset. 

![mnist_image_github](https://github.com/user-attachments/assets/e932d96d-ca1a-4056-8ffa-562cc911da86)

![NORMAL-3767173-12](https://github.com/user-attachments/assets/8a2d0efb-f691-4823-b62f-6f0377887c9c)

The main purpose of each dataset was to test the contrastive learning model in different domains to ensure it would work, no matter the complexity of the data, where MNIST represents the least complex and the 3-D MRIs are the most complex. 

## Contrastive Learning Framework

What is contrastive learning? At a high level, we are creating labels of our own, and then training our model based on these implicit labels. But, how do we create labels out of nowhere? We use data augmentations: PyTorch and MONAI's transform functions took care of this.

![Screenshot 2024-07-23 at 1 11 05 PM](https://github.com/user-attachments/assets/1a6c0f36-bb21-495c-9e84-32c0c7bc5994)

*Caption: Diagram from the original SimCLR paper showing possible data augmentations on a dog (Ting et al.)*

The logic is as follows: if we take an original image and augment it in some way, we get a new version of the original image that is still linked to the original. If we do two different augmentations, we have two new versions that are linked by their common original image. We call these two augmented images *positive pairs*. And, imagining that we start out with an original dataset of N images, if we follow this process we have 2N augmented images. Each of these augmented images has one corresponding positive pair, but more importantly, compared to the other augmented images, it is a *negative pair*. 

The goal of the contrastive learning model is to pull positive pairs closer and push negative pairs farther apart in the latent space. Why would this be a good learning objective? Because, if the model can learn the most important details of an image, it can recognize that positive pairs are similar, even if there are cutouts, cropping, or color distortions. And, by the learning the most important aspects of an image, we produce a good pretrained model that should perform well for further downstream tasks like classification, etc.

But, we're getting ahead of ourselves: how are we encoding our augmented images into the latent space in the first place? What model are we using?

## Model Architecture

I ended up choosing a vanilla Convolutional Neural Network (CNN) with 2 convolutional layers and a MLP classification head with ReLU nonlinearity. Max pooling was used after each convolutional layer and global pooling was used right before the MLP head. For more complex datasets, a ResNet-18 or ResNet-34 can be used in place of the vanilla CNN. 

The following is the overall contrastive learning framework. The Encoder in my case is the CNN model, and we see that we end up taking the Encoder's learned representations for the downstream tasks, which could be outcome classification or other problems we are using our pretrained model for. The importance of the MLP head is to help us calculate the loss of the network and backpropagate the weights.

![contrastive_learning_framework](https://github.com/user-attachments/assets/d87e97c4-5172-446a-9695-6aa7f8c4d923)

## Loss Function

Before we go into the loss function, it's important to understand how we are judging how similar two augmented images are——we are using cosine similarity.

![oreilly_cos_sim](https://github.com/user-attachments/assets/68ea02ef-f6a9-4156-bc52-65d68aac020f)

*The angle between two vectors represents how similar they are: cos(0°) = 1 for most similar, versus cos(180°) = -1 for most dissimilar (Image Source: Statistics for Machine Learning by Pratap Dangeti)*

The loss is calculated by summing up all the losses for all positive pairs. Now, let's take a look at the loss function calculation for one positive pair.

![contrastive_loss](https://github.com/user-attachments/assets/7b2745c6-8bf8-4e2f-a583-c3239a82a806)

The contrastive loss here used to update the model's weights seems complex, but it's actually relatively simple to understand if we break it up into two parts: the numerator and denominator. 

In words, what the numerator is calculating is the similarity between the positive pairs in the latent space, which is represented by z. We'll get to the temperature term, τ, soon.

The denominator is summing up the similarities between the positive pair and all negative pairs. 

Now, let's now work backwards and understand why these two terms work together to help us train our model. Firstly, we see a log term, and immediately we can recognize that as our fraction gets closer to 1, we get closer to a loss of -log(1) = 0. 

How does this loss approach 0? Well, it's important to note that the only term present in both the numerator and denominator is the exponentiated cosine similarity of the positive pair. From calculus, we know if one term is present in both the numerator and denominator, and it dominates the rest of the terms in the fraction, the fraction approaches 1! 

In other words, if the positive pair's similarity is considerably larger than the similarity of the negative pairs, our model will achieve a loss closer to 0, and this makes sense because this is what we want our model to do: bring the positive pairs closer and the negative pairs further apart. 

Of course, the overall loss term will just sum up all the loss calculations for all positive pairs, and divide by a normalizing term——and that's it!

Now, onto the temperature term, a hyperparameter which is pretty cool in my opinion. This is mostly a side-note, since in many implementations, including mine, we set the temperature term to 1, so it isn't relevant. So, when do we play around with this hyperparameter?

Consider when τ = 1. After the cosine similarity calculations, we know that the similarity between any two augmented images has a range of -1 ≤ sim(i,j) ≤ 1, so the numerator would have a range of e^-1 ≤ exp(sim(i,j)) ≤ e. 

Now, if negative pairs are 90° apart, this is still e^0 = 1, which is not a small enough term in the denominator, even if the positive pair value = e. Therefore, the loss will still be high, even though there is already decent separation between the vectors. This will push all cases of negative pairs to try to become antiparallel, but this may not be ideal for the dataset the model is trained on

To fix this, τ can be set to a lower value. For example, when τ = 0.1, the numerator now ranges from e^-1 / 0.1 ≤ sim(i,j) ≤ e / 0.1, or simplified to: e^-10 ≤ sim(i,j) ≤ e^10. Now, a separation of 90° in the latent space for the negative pair is more than enough to lower the loss significantly. This is just one case of when it makes sense to tune τ. For another perspective of τ, Geoffrey Hinton states that setting τ to higher values leads to softer probability distributions.

## Training Loop / DataLoader

It's a relatively vanilla PyTorch training loop. With the guidance of my supervisor/principal investigator, I made 4 functions: predict, train_one_step, train_one_epoch, and train. Each module builds upon the other, and I call the train function in my main.py to train the network. I used 

The DataLoader is responsible for taking in all N images of the dataset, producing two random augmentations of each image, and most importantly keeping track of positive and negative pairs, which is necessary for the training loop functions.

Feel free to check out the code for more detailed documentation surrounding the training loop and dataloader. 

## Overfit Test

With the model, loss, dataloader, and training loop defined, I could now conduct overfit tests on the dataset. Overfit tests are simply when you test out a model on toy datasets, which could either be very simple ones like MNIST or more complex ones like MRIs——just with fewer data samples so that the model can overfit and "learn" (memorize) the training dataset. This is important because if the model cannot succeed on such simple datasets, it will never work in real-world cases.

I started with the MNIST dataset. One of my earliest runs was using an MNIST dataset of only 50 images, with a batch size of 10, and 1000 epochs. Here is the associated loss curve:

![mNist_batch=10_1000epochs](https://github.com/user-attachments/assets/cacd546d-19d2-4dc5-9dc3-02393eface34)

As we can see, the loss starts at around 2.95 and decreases as we reach epoch 1000. However, this decrease has high variance, so it's clear that the contrastive learning framework is not performing well. The key reason for this is a fact that the authors of SimCLR found: the framework does not work well on smaller datasets. So, I tried a larger MNIST dataset of 10,000 images, with a batch size of 50, and now only 100 epochs.

![mnistLarge_batch=50_epoch=100](https://github.com/user-attachments/assets/2b61ebed-87e6-42e8-aae4-12ecf46bd7a1)

In this case, we can see that the model learns much quicker, taking only 100 epochs to decrease a loss of 4.6 to almost 4. It gives credence to the idea that larger datasets and larger batch sizes tend to benefit model training for contrastive learning, which was found by Ting et al. in the original SimCLR paper. 

Now, it was time to test the model on an actual healthcare-related dataset, such as brain MRIs or OCT scans. For brevity, I will only show the results for the MRIs.

The MRIs are 3-D, so I had to change the 2D convolutions in the model to 3D convolutions. I used 4 brain MRIs for the overfit test, and I first downsampled the MRIs to 75x75x75, and then trained the model for 1000 epochs.

![1000Epochs_4MRIs_75x75x75](https://github.com/user-attachments/assets/65657598-d14e-4515-a655-267b593211c2)

As can be seen, the model did not train at all, which worried me. I decided that it was possible the model simply struggled since the data was too high-dimensional, so I downsampled the MRIs to 50x50x50 and ran the training again.

![1000epochs_4MRIs_50x50x50](https://github.com/user-attachments/assets/bc238f1d-f807-416b-93bf-6a543721aafe)

In this simpler case, the model is able to overfit after only 75 epochs, going from an initial loss of 1.08 to 0.24. 

## Conclusion and Future Direction

My summer project was overall a success: I was able to successfully conduct overfit tests on MNIST and a brain MRI dataset, verifying the findings of the SimCLR paper and providing a contrastive learning framework for the Chiang Lab for future projects to use.

With more time, it's clear the model will have to be made more complex, by using a ResNet or perhaps even Vision Transformer backbone. The model did well on simpler datasets, but struggled on harder, higher dimensional images. With more tuning however, it's very promising for this framework to pretrain models that can be used to benefit the healthcare system exponentially.

## Appendix

In case you wanted to read more about my project, here is a [paper](https://docs.google.com/document/d/1e7BlJMWSNGH-GAMD3SJdu0gE33xDemdk5tXxXMNTdRc/edit?usp=sharing) I wrote on it and the associated [presentation](https://docs.google.com/presentation/d/1QQIuXgCDy5wjsRplh_0wCN7_dRoeOS-zHy1-D5eqhl0/edit?usp=sharing) I gave at UCLA as my departmental thesis.





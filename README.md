# Image Recognition
Image Recognition Implementation Using Logistic Regression

Some of this was also used in Tom Fletcher's CS3501 *Foundations of Data Analysis* in a submission for Assignment #5

This program can identify two different images (and only two...) given all image values are between (0, 255). In this repository is also a method of converting CIFAR Images to the correct format. 

## How to Use

First, the testing functionality uses [Jupyter](https://jupyter.org/) for some graphing/image functionality. [Here](https://jupyter.org/install) is a link for installation. Once downloaded, run 

`pip install -r requirements.txt`

Then, download the CIFAR Datasets for Python from [here](https://www.cs.toronto.edu/~kriz/cifar.html). Either CIFAR dataset works, but this is inteded to use the CIFAR-10, but CIFAR-100 should also work!

Here is some info on CIFAR: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

Once completed, just mess around with the values in test.ipynb (or don't) and view the results.

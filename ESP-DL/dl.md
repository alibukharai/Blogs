# Building with ESP-DL 

Artificial intelligence transform the way computer interact with real world. Decision are carried by getting data from Tiny low powered devices and sensors into the cloud. Connectivity, high cost and data privacy are some of the demerits of this method. Edge artificial intelligence is another way to process the data right on the physical device without sending data back and forth improving the latency and security and reduced the  bandwidth and power.


[Espressif System](https://www.espressif.com/) provides a new library [ESP-DL](https://github.com/espressif/esp-dl) that can be used to deploy your high performance deep learning model right at the top of [ESP32-S3](https://www.espressif.com/en/products/socs/esp32-s3). 

*In this article, we will understand how to use [ESP-DL](https://github.com/espressif/esp-dl) and [deploy](https://github.com/espressif/esp-dl/tree/master/tutorial/quantization_tool_example) a deep learning model on [ESP32-S3](https://www.espressif.com/en/products/socs/esp32-s3).*

---
# Content 
The article is divided into 3 portions
1. Model development<sup>[link](#1-model-development)  
2. ESP-DL formate Conversion <sup>[link](#2-esp-dl-formate) 
3. Model Deployment<sup>[link](#3-model-deployment)  
---

## Getting Started with ESP-DL
* Before getting a deep dive into ESP-DL, I assume that readers have;  

    1. Knowledge to build and train their own neural networks.<sup>  [video](https://www.youtube.com/watch?v=WvoLTXIjBYU) 
    2. Configure the ESP-IDF environment. <sup>[video](https://www.youtube.com/watch?v=byVPAfodTyY) / [blog](https://blog.espressif.com/esp-idf-development-tools-guide-part-i-89af441585b) 
    3. Working knowledge of basic C language.<sup>[video](https://www.youtube.com/watch?v=KJgsSFOSQv0&t=12665s)


### 1. Model Development
For the sake of simplicity I am using a classification problem and developed a simple deep learning model to classifying 6 different hand gestures. Many pre-trained [models](https://github.com/filipefborba/HandRecognition) are also available however I prefer to build my own to get better understanding of each layer of model.\
<sup> * Model is trained in Google [Co-lab](https://colab.research.google.com/) 


#### 1.1 Dataset

For this classification problem I am using an open source dataset from kaggle [Hand Gesture recognition Dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog). The original dataset includes 10 classes however I am using only 6 classes that is easy to recognize and more useful in daily life. The hand gesture classes are represented in the table below. One more difference is related to the image size, The original dataset have an image size of (240,640) however for the sake of simplicity we resized the dataset to (96,96). The dataset used in this article can be found from [here]().

<p align = 'center'>

| Gesture     | Label Used |
| :-----------: | :-----------: |
| Palm        |   0         |
| I           |   1         |
|Thumb        |   2         |
|Index        |   3         |
|OK           |   4         |
|C            |   5         |
<p align = 'center'> <b>Table 1 — Classification used for every hand gesture.</b>

<p align = 'left'>

#### 1.2 Test/Train Split 

We need to divide our dataset into test, train and calibration dataset. These datasets are nothings but the subsets of our original [dataset](#11-dataset). Train datset is used to train the model while testing datset is to test the model performance similarly calibration datset is used during the [model quantization]() stage for calibration. Procedure to generate all these datasets are the same.

```
from sklearn.model_selection import train_test_split

ts = 0.3 # Percentage of images that we want to use for testing. 
X_train, X_test1, y_train, y_test1 = train_test_split(X, y, test_size=ts, random_state=42)
X_test, X_cal, y_test, y_cal = train_test_split(X_test1, y_test1, test_size=ts, random_state=42)

```
<sub> For more details about how train_test_split works please check [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

For the sake of simplicity and reproduction of this tutorial I also provided the data used in Pickel file formate. use the below block of code to open data in your environment.   

```
import pickle

with open('X_test.pkl', 'rb') as file:
    X_test = pickle.load(file)

with open('y_test.pkl', 'rb') as file:
    y_test = pickle.load(file)

with open('X_train.pkl', 'rb') as file:
    X_train = pickle.load(file)

with open('y_train.pkl', 'rb') as file:
    y_train = pickle.load(file)
```

#### 1.3 Creating a Model

I have created a basic Convoltion Neural Network (CNN)for this classification problem. It consisists of 3 convolution layer followed by maxpooling and fully connected layer with an output of 6 neurons.  For more details about the creation of [CNN](https://github.com/filipefborba/HandRecognition/blob/master/project3/project3.ipynb). Below is the code used to create a CNN. 

```
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
print(tf.__version__)

model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(96, 96, 1))) 
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
```
<p align="center">
    <img src="./1.png#center">

#### 1.4 Training Model
CNN is trained for 5 epoches and its gives and accuracy of around 99%. 
```
history=model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1, validation_data=(X_test, y_test))
```

<p align="center">
    <img src="./2.png#center">

#### 1.5 Saving Model 
The trained model is saved in .h5 formate. 

``` 
model.save('handrecognition_model.h5')
```
#### 1.6 Model Conversion
ESP-DL uses model in ONXX [formate](https://onnx.ai/). To be competible with ESP-DL I transformed a model into ONXX by using below lines of code. 

```
model = tf.keras.models.load_model("/content/handrecognition_model.h5")
tf.saved_model.save(model, "tmp_model")
!python -m tf2onnx.convert --saved-model tmp_model --output "handrecognition_model.onnx"
!zip -r /content/tmp_model.zip /content/tmp_model
```
Finally downloaded .h5 model, ONNX model and model checkpoints. 

```
from google.colab import files
files.download("/content/handrecognition_model.h5")
files.download("/content/handrecognition_model.onnx")
files.download("/content/tmp_model.zip")
```

### 2. ESP-DL Formate
Once the ONNX model is ready, by using the steps below to convert the model into ESP-DL formate.\
<sup> *[Pychram](https://www.jetbrains.com/pycharm/) IDE is used to convert model in ESP-DL formate.  

### 1.1 Requirements <sup>[link](https://github.com/espressif/esp-dl/tree/master/tutorial/quantization_tool_example#step-121-set-up-the-environment) 
Setting up an environment and installing correct verison of the modules is a key to start with. If the modules are not installed in correct version it produces an error. 

<p align = 'center'>

| Module     | How to install  |
| :-----------: | :-----------: |
| Python == 3.7        |   -        |
| Numba == 0.53.1        |   `pip install Numba==0.53.1`           |
| ONNX == 1.9.0           |   `pip install ONNX==1.9.0`        |
|ONNX Runtime == 1.7.0       |  `pip install ONNXRuntime==1.7.0`         |
|ONNX Optimizer == 0.2.6            |  `pip install ONNXOptimizer==0.2.6`        |
| 
<p align = 'center'>
<b>Table 2 — Required Modules and specified verisons </b>.
<p align = 'left'>

Second thing we need to clone the [ESP-DL](https://github.com/espressif/esp-dl) from the github.  
```
git clone -- recursive https://github.com/espressif/esp-dl.git
```
### 1.2 Optimization and Quantization 
To run the optimizer provided by ESP-DL,  we need to find and  
- calibrator.pyd
- calibrator_acc.pyd
- evaluator.pyd 
- optimizer.py

place these files into the working directory of pychram. Furthermore also place the calibration dataset generated in previous [section](#12-testtrain-split) and ONNX model saved in previous [section](#15-saving-model). Your working directory looks like this;

<p align="center">
    <img src="./3.png#center">

Follow the below steps for generating optimzed and quantized model. 

1. import the libraries 

```
from optimizer import *
from calibrator import *
from evaluator import *
```

2. Load the ONNX Model 

```
onnx_model = onnx.load("handrecognition_model.onnx")
```

3. Optimize the ONNX model 

```
optimized_model_path = optimize_fp_model("handrecognition_model.onnx")
```
4. Load Calibration dataset 
```
with open('X_cal.pkl', 'rb') as f:
    (test_images) = pickle.load(f)
with open('y_cal.pkl', 'rb') as f:
    (test_labels) = pickle.load(f)


calib_dataset = test_images[0:1800:20]
pickle_file_path = 'handrecognition_calib.pickle'
```
5. Calibration 
```
model_proto = onnx.load(optimized_model_path)
print('Generating the quantization table:')

calib = Calibrator('int16', 'per-tensor', 'minmax')
# calib = Calibrator('int8', 'per-channel', 'minmax')

calib.set_providers(['CPUExecutionProvider'])

# Obtain the quantization parameter
calib.generate_quantization_table(model_proto,calib_dataset, pickle_file_path)
# Generate the coefficient files for esp32s3
calib.export_coefficient_to_cpp(model_proto,  pickle_file_path, 'esp32s3', '.', 'handrecognition_coefficient', True)

```
Two new files with extension .cpp and .hpp is generated in the path, and output looks like this. 

<sup> *Take screenshot of this output later it will be used. 


<p align="center">
    <img src="./4.png#center">

### 1.3 Evaluate 
This step is not necessary however if you want to see the performance of optimized model the following code can be run. 
```
print('Evaluating the performance on esp32s3:')
eva = Evaluator('int16', 'per-tensor', 'esp32s3')
eva.set_providers(['CPUExecutionProvider'])
eva.generate_quantized_model(model_proto, pickle_file_path)

output_names = [n.name for n in model_proto.graph.output]
providers = ['CPUExecutionProvider']
m = rt.InferenceSession(optimized_model_path, providers=providers)

batch_size = 64
batch_num = int(len(test_images) / batch_size)
res = 0
fp_res = 0
input_name = m.get_inputs()[0].name
for i in range(batch_num):
    # int8_model
    [outputs, _] = eva.evalute_quantized_model(test_images[i * batch_size:(i + 1) * batch_size], False)
    res = res + sum(np.argmax(outputs[0], axis=1) == test_labels[i * batch_size:(i + 1) * batch_size])

    # floating-point model
    fp_outputs = m.run(output_names, {input_name: test_images[i * batch_size:(i + 1) * batch_size].astype(np.float32)})
    fp_res = fp_res + sum(np.argmax(fp_outputs[0], axis=1) == test_labels[i * batch_size:(i + 1) * batch_size])
print('accuracy of int8 model is: %f' % (res / len(test_images)))
print('accuracy of fp32 model is: %f' % (fp_res / len(test_images)))

```

## 3. Model Deployment  
1. Building in ESP-IDF
2. 



### Running a Model 




### Output 



## Future 
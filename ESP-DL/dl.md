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


#### 1.1. Dataset

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

#### 1.2. Test/Train Split 

We need to divide our dataset into test, train and calibration dataset. These datasets are nothings but the subsets of our original [dataset](#11-dataset). Train datset is used to train the model while testing datset is to test the model performance similarly calibration datset is used during the [model quantization]() stage for calibration. Procedure to generate all these datasets are the same.

```
from sklearn.model_selection import train_test_split

ts = 0.3 # Percentage of images that we want to use for testing. 
X_train, X_test1, y_train, y_test1 = train_test_split(X, y, test_size=ts, random_state=42)
X_test, X_cal, y_test, y_cal = train_test_split(X_test1, y_test1, test_size=ts, random_state=42)

```
<sub> For more details about how train_test_split works please check [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

For the sake of simplicity and reproduction of this tutorial I also provided the data used in Pickel file formate. use the below block of code to open data in your environment.   

```python
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

#### 1.3. Creating a Model

I have created a basic Convoltion Neural Network (CNN)for this classification problem. It consisists of 3 convolution layer followed by maxpooling and fully connected layer with an output of 6 neurons.  For more details about the creation of [CNN](https://github.com/filipefborba/HandRecognition/blob/master/project3/project3.ipynb). Below is the code used to create a CNN. 

```python
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

#### 1.4. Training Model
CNN is trained for 5 epoches and its gives and accuracy of around 99%. 

```python 
history=model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1, validation_data=(X_test, y_test))
```

<p align="center">
    <img src="./2.png#center">

#### 1.5. Saving Model 
The trained model is saved in .h5 formate. 

```python 
model.save('handrecognition_model.h5')
```
#### 1.6. Model Conversion
ESP-DL uses model in ONXX [formate](https://onnx.ai/). To be competible with ESP-DL I transformed a model into ONXX by using below lines of code. 

```python
model = tf.keras.models.load_model("/content/handrecognition_model.h5")
tf.saved_model.save(model, "tmp_model")
!python -m tf2onnx.convert --saved-model tmp_model --output "handrecognition_model.onnx"
!zip -r /content/tmp_model.zip /content/tmp_model
```
Finally downloaded .h5 model, ONNX model and model checkpoints. 

```python
from google.colab import files
files.download("/content/handrecognition_model.h5")
files.download("/content/handrecognition_model.onnx")
files.download("/content/tmp_model.zip")
```

### 2. ESP-DL Formate
Once the ONNX model is ready, by using the steps below to convert the model into ESP-DL formate.\
<sup> *[Pychram](https://www.jetbrains.com/pycharm/) IDE is used to convert model in ESP-DL formate.  

### 2.1. Requirements <sup>[link](https://github.com/espressif/esp-dl/tree/master/tutorial/quantization_tool_example#step-121-set-up-the-environment) 
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
### 2.2. Optimization and Quantization 
To run the optimizer provided by ESP-DL,  we need to find and  
- calibrator.pyd
- calibrator_acc.pyd
- evaluator.pyd 
- optimizer.py

place these files into the working directory of pychram. Furthermore also place the calibration dataset generated in previous [section](#12-testtrain-split) and ONNX model saved in previous [section](#15-saving-model). Your working directory looks like this;

<p align="center">
    <img src="./3.png#center">

Follow the below steps for generating optimzed and quantized model. 

#### 2.2.1. import the libraries 

```python
from optimizer import *
from calibrator import *
from evaluator import *
```

#### 2.2.2. Load the ONNX Model 

```python
onnx_model = onnx.load("handrecognition_model.onnx")
```

#### 2.2.3. Optimize the ONNX model 

```python
optimized_model_path = optimize_fp_model("handrecognition_model.onnx")
```
#### 2.2.4. Load Calibration dataset 
```python
with open('X_cal.pkl', 'rb') as f:
    (test_images) = pickle.load(f)
with open('y_cal.pkl', 'rb') as f:
    (test_labels) = pickle.load(f)


calib_dataset = test_images[0:1800:20]
pickle_file_path = 'handrecognition_calib.pickle'
```
#### 2.2.5. Calibration 
```python
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


### 2.3. Evaluate 
This step is not necessary however if you want to see the performance of optimized model the following code can be run. 

```python
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
<sup>* please follow the [link](https://github.com/espressif/esp-dl/tree/master/tutorial/quantization_tool_example#step-122-optimize-your-model) for more details 

## 3. Model Deployment  
### 3.1. ESP-IDF Project Hirarchy

The first step is to create a new project in VS-Code based on ESP-IDF standards. <sup>[video](https://www.youtube.com/watch?v=Lc6ausiKvQM) / [blog](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/linux-macos-setup.html)</sup> and copy the files.cpp and .hpp generated in the previous [section](#12-optimization-and-quantization) to your current working directory. Add components folder to your working directory. The Project directory should looks like the picture below;

<p align="center">
    <img src="./5.png#center">

### 3.2. Model define

#### 3.2.1. Import libraries

```c
#pragma once
#include <stdint.h>
#include "dl_layer_model.hpp"
#include "dl_layer_base.hpp"
#include "dl_layer_max_pool2d.hpp"
#include "dl_layer_conv2d.hpp"
#include "dl_layer_concat.hpp"
#include "handrecognition_coefficient.hpp"
#include "dl_layer_reshape.hpp"
#include "dl_layer_softmax.hpp"

using namespace dl;
using namespace layer;
using namespace handrecognition_coefficient;
```
#### 3.2.2. Declare layers

```c
class HANDRECOGNITION : public Model<int16_t> 
{
private:
    Reshape<int16_t> l1;
    Conv2D<int16_t> l2;
    MaxPool2D<int16_t> l3;
    Conv2D<int16_t> l4;
    MaxPool2D<int16_t> l5;
    Conv2D<int16_t> l6;
    MaxPool2D<int16_t> l7;
    Reshape<int16_t> l8;
    Conv2D<int16_t> l9;
    Conv2D<int16_t> l10;
public:
    Softmax<int16_t> l11; // output layer 
```

#### 3.2.3. Initialize layers 

```c
 HANDRECOGNITION () : l1(Reshape<int16_t>({96,96,1})),
                         l2(Conv2D<int16_t>(-8, get_statefulpartitionedcall_sequential_1_conv2d_3_biasadd_filter(), get_statefulpartitionedcall_sequential_1_conv2d_3_biasadd_bias(), get_statefulpartitionedcall_sequential_1_conv2d_3_biasadd_activation(), PADDING_VALID, {}, 1,1, "l1")),
                         l3(MaxPool2D<int16_t>({2,2},PADDING_VALID, {}, 2, 2, "l2")),                      
                         l4(Conv2D<int16_t>(-9, get_statefulpartitionedcall_sequential_1_conv2d_4_biasadd_filter(), get_statefulpartitionedcall_sequential_1_conv2d_4_biasadd_bias(), get_statefulpartitionedcall_sequential_1_conv2d_4_biasadd_activation(), PADDING_VALID,{}, 1,1, "l3")),                       
                         l5(MaxPool2D<int16_t>({2,2},PADDING_VALID,{}, 2, 2, "l4")),                       
                         l6(Conv2D<int16_t>(-9, get_statefulpartitionedcall_sequential_1_conv2d_5_biasadd_filter(), get_statefulpartitionedcall_sequential_1_conv2d_5_biasadd_bias(), get_statefulpartitionedcall_sequential_1_conv2d_5_biasadd_activation(), PADDING_VALID,{}, 1,1, "l5")),                    
                         l7(MaxPool2D<int16_t>({2,2},PADDING_VALID,{}, 2, 2, "l6")),
                         l8(Reshape<int16_t>({1,1,6400},"l7_reshape")), //16,2id8,28 or 1,12544 or 12544, or 1,1,12544 or 1,16,28,28 or 1,16,28,28
                         l9(Conv2D<int16_t>(-9, get_fused_gemm_0_filter(), get_fused_gemm_0_bias(), get_fused_gemm_0_activation(), PADDING_VALID, {}, 1, 1, "l8")),
                         l10(Conv2D<int16_t>(-9, get_fused_gemm_1_filter(), get_fused_gemm_1_bias(), NULL, PADDING_VALID,{}, 1,1, "l9")),
                         l11(Softmax<int16_t>(-14,"l10")){}
```
#### 3.2.4. Build layers


```c
void build(Tensor<int16_t> &input)
    {
        this->l1.build(input);
        this->l2.build(this->l1.get_output());
        this->l3.build(this->l2.get_output());
        this->l4.build(this->l3.get_output());
        this->l5.build(this->l4.get_output());
        this->l6.build(this->l5.get_output());
        this->l7.build(this->l6.get_output());
        this->l8.build(this->l7.get_output());
        this->l9.build(this->l8.get_output());
        this->l10.build(this->l9.get_output());
        this->l11.build(this->l10.get_output());        
    }
```
#### 3.2.5. Call layers

```c
void call(Tensor<int16_t> &input)
    {
        this->l1.call(input);
        input.free_element();

        this->l2.call(this->l1.get_output());
        this->l1.get_output().free_element();

        this->l3.call(this->l2.get_output());
        this->l2.get_output().free_element();

        this->l4.call(this->l3.get_output());
        this->l3.get_output().free_element();

        this->l5.call(this->l4.get_output());
        this->l4.get_output().free_element();

        this->l6.call(this->l5.get_output());
        this->l5.get_output().free_element();

        this->l7.call(this->l6.get_output());
        this->l6.get_output().free_element();

        this->l8.call(this->l7.get_output());
        this->l7.get_output().free_element();

        this->l9.call(this->l8.get_output());
        this->l8.get_output().free_element();

        this->l10.call(this->l9.get_output());
        this->l9.get_output().free_element();

        this->l11.call(this->l10.get_output());
        this->l10.get_output().free_element();
    }
};
```
### 3.3. Model Run

#### 3.3.1. import libraries
```c
#include <stdio.h>
#include <stdlib.h>
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "dl_tool.hpp"
#include "model_define.hpp"
```

#### 3.3.2. Declare Input 
```cpp
int input_height = 96;
int input_width = 96;
int input_channel = 1;
int input_exponent = -7;

__attribute__((aligned(16))) int16_t example_element[] = {

    //add your input image pixels 
};
```

#### 3.3.3. Set Input Shape

```cpp
extern "C" void app_main(void)
{
for (int i=0; i<9216;i++){

    // printf("0x%02x,",example_element[i]);
}
Tensor<int16_t> input;
                input.set_element((int16_t *)example_element).set_exponent(input_exponent).set_shape({input_height,input_width,input_channel}).set_auto_free(false);
```

#### 3.3.4. Call Model 

```cpp
HANDRECOGNITION model;
                dl::tool::Latency latency;
                latency.start();
                model.forward(input);
                latency.end();
                latency.print("\nSIGN", "forward");
```

### 3.3.5. Monitor Output 

```c
float *score = model.l11.get_output().get_element_ptr();
                float max_score = score[0];
                int max_index = 0;
                for (size_t i = 0; i < 6; i++)
                {
                    printf("%f, ", score[i]*100);
                    if (score[i] > max_score)
                    {
                        max_score = score[i];
                        max_index = i;
                    }
                }
                printf("\n");

                switch (max_index)
                {
                    case 0:
                    printf("Palm: 0");
                    break;
                    case 1:
                    printf("I: 1");
                    break;
                    case 2:
                    printf("Thumb: 2");
                    break;
                    case 3:
                    printf("Index: 3");
                    break;
                    case 4:
                    printf("ok: 4");
                    break;
                    case 5:
                    printf("C: 5");
                    break;
                    default:
                    printf("No result");
                }
                printf("\n");

}
```
## 4. Future Work 

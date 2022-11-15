# Building with ESP-DL 

Artificial intelligence transform the way computer interact with real world. Decision are carried by getting data from Tiny low powered devices and sensors into the cloud. Connectivity, high cost and data privacy are some of the demerits of this method. Edge artificial intelligence is another way to process the data right on the physical device without sending data back and forth improving the latency and security and reduced the  bandwidth and power.


[Espressif System](https://www.espressif.com/) provides a new library [ESP-DL](https://github.com/espressif/esp-dl) that can be used to deploy your high performance deep learning model right at the top of [ESP32-S3](https://www.espressif.com/en/products/socs/esp32-s3). 

*In this article, we will understand how to use [ESP-DL](https://github.com/espressif/esp-dl) and [deploy](https://github.com/espressif/esp-dl/tree/master/tutorial/quantization_tool_example) a deep learning model on [ESP32-S3](https://www.espressif.com/en/products/socs/esp32-s3).*


## Getting Started with ESP-DL
Before getting a deep dive into ESP-DL, I assume that readers have;  

1. Knowledge to build and train their own neural networks.<sup>  [video](https://www.youtube.com/watch?v=WvoLTXIjBYU) 
2. Configure the ESP-IDF environment. <sup>[video](https://www.youtube.com/watch?v=byVPAfodTyY) / [blog](https://blog.espressif.com/esp-idf-development-tools-guide-part-i-89af441585b) 
3. Working knowledge of basic C language.<sup>[video](https://www.youtube.com/watch?v=KJgsSFOSQv0&t=12665s)

### Model Development 
For the sake of simplicity I am using a classification problem and developed a simple deep learning model to classifying 6 different hand gestures. Many pre-trained [models](https://github.com/filipefborba/HandRecognition) are also available however I prefer to build my own to get better understanding of each layer of model.  

1. Dataset
For this classification problem I am using an open source dataset from kaggle [Hand Gesture recognition Dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog). The original dataset includes 10 classes however I am using only 6 classes that is easy to recognize and more useful in daily life. The hand gesture classes are represented in the table below. The dataset used in this article can be found from [here](). 

<br /> 
<p align = 'center'>

| Gesture     | Label Used |
| :-----------: | :-----------: |
| Palm        |   0         |
| I           |   1         |
|Thumb        |   2         |
|Index        |   3         |
|OK           |   4         |
|C            |   5         |
<p align = 'center'>
<b>Table 1 — Classification used for every hand gesture.
<p align = 'left'>

2. Test/Train Split 




```
ts = 0.3 # Percentage of images that we want to use for testing. The rest is used for training.
X_train, X_test1, y_train, y_test1 = train_test_split(X, y, test_size=ts, random_state=42)
X_test, X_cal, y_test, y_cal = train_test_split(X_test1, y_test1, test_size=ts, random_state=42)

```

[link](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)



3. Model Designing 


```



```

4. Training Model

```


```

5. Saving Model 


``` 



```

### Model Conversion
1. To ONNX formate

```

```


2. To ESP-DL formate

    1. Requirement 
    2. Optimization
    3. Quantization 


### Model Deployment  
1. Building in ESP-IDF
2.  



### Running a Model 




### Output 



## Future 
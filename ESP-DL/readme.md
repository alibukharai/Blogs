# Steps to download and run 
Before follow these steps please Configure the ESP-IDF environment. <sup>[setting-up ESP-IDF environment](https://www.youtube.com/watch?v=byVPAfodTyY) / [toolchain for ESP-IDF](https://blog.espressif.com/esp-idf-development-tools-guide-part-i-89af441585b) 

## 1. Clone the git-hub repository 

```bash
git clone https://github.com/alibukharai/Blogs.git 

```
## 2. Update the submodules

```bash
git submodule update --init --recursive 

```

## 3. Change the working directory to model_deployment

```bash 
cd ESP-DL/model_deployment

```

## 4. Reconfigure the Cmake 

```bash 
idf.py reconfigure 

```

## 5. Select the target ESP32

```bash 
idf.py set-target esp32s3

```
## 6. Make sure the ESP32-S3 connected to the PC

```bash
idf.py build flash monitor

```
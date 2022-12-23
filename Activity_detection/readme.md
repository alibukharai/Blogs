# **Steps to download and run** 
Before follow these steps please Configure the ESP-IDF [release 4.4](https://github.com/espressif/esp-idf/tree/release/v4.4) environment. <sup>[setting-up ESP-IDF environment](https://www.youtube.com/watch?v=byVPAfodTyY) / [toolchain for ESP-IDF](https://blog.espressif.com/esp-idf-development-tools-guide-part-i-89af441585b) 

### 1. Clone the git-hub repository 

```bash
git clone https://github.com/alibukharai/Blogs.git 

```
### 2. Update the submodules

```bash
git submodule update --init --recursive 

```

### 3. Change the working directory to model_deployment

```bash 
cd Activity_detection/model_deployment

```

### 4. Reconfigure the Cmake 

```bash 
idf.py reconfigure 

```

### 5. Select the target ESP32

```bash 
idf.py set-target esp32s3

```
### 6. Build the project

```bash
idf.py build flash monitor 

```
In case found error during the building process [follow the official IDF  guide for more details](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html#build-your-first-project). 
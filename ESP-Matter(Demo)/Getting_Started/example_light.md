# **Step by step understanding of the ESP-MATTER**
Matter or Project Connected Home over IP (CHIP) is a open source smart home connectivity standard. It enables secure and reliable connectivity among different smart home and building devices from different vendors over internet protocol(IP). <br>

- The first version of the [specification](https://csa-iot.org/all-solutions/matter/) is already out for the developers to use. This includes some features like mains power plugs, electric lights and switches), door locks, thermostats and heating, ventilation, and air conditioning controllers, blinds and shades, home security sensors (such as door, window and motion sensors), and televisions and streaming video players.
- The second version planned to include support for   robotic vacuum cleaners, ambient motion and presence sensing, smoke and carbon monoxide detectors, environmental sensing and controls, closure sensors, energy management, Wi-Fi access points, cameras and major appliances.More information about the matter can be found [here](https://www.consumerreports.org/smart-home/matter-smart-home-standard-faq-a9475777045) <br>

*[Espressif](https://www.espressif.com/en) provide the complete [SDK](https://docs.espressif.com/projects/esp-matter/en/main/esp32/index.html) to the IoT developers to integrate Matter to the Espressif [SOCs](https://www.espressif.com/en/products/socs).* 

The motivation of writing this article is to give a quick overview of starting and building with Espressif Matter SDK 

## Requirements 

- ESP-IDF [release 4.4](https://github.com/espressif/esp-idf/tree/release/v4.4)
- ESP-[Matter](https://github.com/espressif/esp-matter)

<Sup>A Quick installation guide of [ESP-IDF](https://blog.espressif.com/esp-idf-development-tools-guide-part-i-89af441585b) and [ESP-Matter](https://github.com/alibukharai/Blogs/blob/main/ESP-Matter(Demo)/Getting_Started/getting_started_with_matter_esp.md) 


## 1 Building an example 
Choose an example from esp-matter folder and then follow the steps below. Here in this article we are working with [light example](https://github.com/espressif/esp-matter/tree/main/examples/light).

### 1.1 Export ESP-IDF release 4.4 into the terminal by using
Goto the ESP-IDF installation folder and run;  

```bash
. /export.sh

```
### 1.2 Export ESP-Matter into the terminal by using 
Goto the ESP-Matter Folder and run;
```bash
. ./export.sh

```

### 1.3 Select the device 
I am using ESP32 C3, however you can check the supported hardware based on your application 

```bash 
idf.py select-target esp32c3

```

### 1.4 Build and Run
Change the directory to the light folder under ESP-Matter/examples and run; 
```bash 
idf.py build flash monitor

```
Until now the everything is simple and if everything is right the light on the board turn on you can also control the status of the LED from the boot button on the board. The output should looks like this; 

<p align="center">
    <img src="./_static/1 output.png#center">


From this output you can see that the commissioning window (mark in red box) is opened. In simple words your device is ready to get communicate with outside world (external devices).


## 2 Commissioning 
There are [several ways](https://github.com/project-chip/connectedhomeip/blob/master/docs/guides/esp32/build_app_and_commission.md#commissioning) to commission and control the device. In this tutorial we are using standalone chip-tool app to commission and control the device. 

*it is a good practice to open and split the terminal so when you send a command you can monitor the device output*

### 2.1 Building Standalone chip-tool

```bash 
cd path/to/connectedhomeip
scripts/examples/gn_build_example.sh examples/chip-tool out/debug

```

<p align="center">
    <img src="./_static/1 output.png#center">
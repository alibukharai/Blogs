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
To build the Standalone  chip-tool follow the below block of code. For more information about [stand alone chip-tool](https://github.com/project-chip/connectedhomeip/tree/master/examples/chip-tool).

```bash 
cd path/to/connectedhomeip
scripts/examples/gn_build_example.sh examples/chip-tool out/debug

``` 
<p align="center">
    <img src="./_static/2.png#center">

### 2.2 Commission a device 

To commissions a device we will send a request to our device using chip-tool

```bash
out/debug/chip-tool pairing ble-wifi 123 SSID Password 20202021 3840

```

- 123: Node id you can choose any node id for an example 12345, 0x7283 etc. 
- SSID: SSID of the Wifi
- Password: Password of your wifi network
- 20202021: Setup Pin Code
- 3840: Discriminator <br>
*To assign different Setup Pin Code & Discriminator [Follow here](https://github.com/project-chip/connectedhomeip/blob/master/docs/guides/esp32/factory_data.md)* 

The output looks like this; on left side(black-white) the chip-tool output, on the right side(black-green) device output 

<p align="center">
    <img src="./_static/3.png#center">


You can see the node id is **node ID: 0x000000000000007B** which is same as 123 in decimal number. Now the device is commissioned successfully. In other words your device becomes a matter node with id 0x7b.

## 3 Control 
To control the light we need to send a command. The command usually in the form of Cluster name + command+ Node ID + End point. 
### 3.1 Toggle OnOff
```
out/debug/chip-tool onoff toggle 0x7B 0x1

```

<p align="center">
    <img src="./_static/4.png#center">

You can see the after receiving the command setting the status of the light from 1(ON) to 0(OFF)

### 3.2 Level Control 

```bash
out/debug/chip-tool levelcontrol move-to-level 10 0 0 0 0x7B 0x1

```
### 3.3 Color Control  
```bash 
out/debug/chip-tool colorcontrol move-to-saturation 200 0 0 0 0x7B 0x1

```

```bash 

out/debug/chip-tool colorcontrol move-to-hue-and-saturation 240 100 0 0 0 0x7B 0x1
```
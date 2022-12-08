

[ESP-Matter](https://blog.espressif.com/matter-clusters-attributes-commands-82b8ec1640a0)


In this article we are using the default [Light example](https://github.com/espressif/esp-matter/tree/main/examples/light) provided by espressif with ESP-Matter SDK.

- Step 1: Create a matter node

To creating a node calling a create function.

```cpp
    node::config_t node_config;
    node_t *node = node::create(&node_config, app_attribute_update_cb, app_identification_cb);

```
The above piece of code creates a matter node with an end point zero. 
Check the implementation of the endpoint zero in [esp_matter_endpoint.cpp](https://github.com/espressif/esp-matter/blob/a0f137865936aa4eac97855de07ca5f5786ffa45/components/esp_matter/esp_matter_endpoint.cpp#L754). The end point zero contains a basic information cluster. To understand more about end point zero [follow here](https://blog.espressif.com/matter-clusters-attributes-commands-82b8ec1640a0). To check the implementation of basic cluster to the root node [follow here](https://github.com/espressif/esp-matter/blob/a0f137865936aa4eac97855de07ca5f5786ffa45/components/esp_matter/esp_matter_endpoint.cpp#L50). 

- Step 1.1: Default values of the LED

This step is not mandatory however it depends on the logic of each developer.  

```cpp
    color_temperature_light::config_t light_config;
    light_config.on_off.on_off = DEFAULT_POWER;
    light_config.on_off.lighting.start_up_on_off = nullptr;
    light_config.level_control.current_level = DEFAULT_BRIGHTNESS;
    light_config.level_control.lighting.start_up_current_level = DEFAULT_BRIGHTNESS;
    light_config.color_control.color_mode = EMBER_ZCL_COLOR_MODE_COLOR_TEMPERATURE;
    light_config.color_control.color_temperature.startup_color_temperature_mireds = nullptr;

```

- Step 2:  Creating an End point 

The next step is to create an end point. This is our End point 1 which represents the LED on the ESP32-C3 dev kit board. Espressif implements a color temperature light in their SDK according to Connectivity Standards Alliance (CSA) Version 1.0. In CSA standard color temperature light (end point) has 5 clusters namely Identity, Groups, Scenes, On/Off, Level Control and Color Control. The implementation can be found [here](https://github.com/espressif/esp-matter/blob/a0f137865936aa4eac97855de07ca5f5786ffa45/components/esp_matter/esp_matter_endpoint.cpp#L144).

```cpp
endpoint_t *endpoint = color_temperature_light::create(node, &light_config, ENDPOINT_FLAG_NONE, light_handle);

```

Step 3: Add attributes to the cluster



```cpp
 cluster_t *cluster = cluster::get(endpoint, ColorControl::Id);

```



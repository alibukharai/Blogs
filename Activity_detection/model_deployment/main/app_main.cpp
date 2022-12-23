#include <stdio.h>
#include <stdlib.h>
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "dl_tool.hpp"
#include "model_define.hpp"
#include "i2c_bus.h"
#include "mpu6050.h"
#include "driver/i2c.h"
#include "esp_log.h"

int input_height = 80;
int input_width = 3;
int input_channel = 1;
int input_exponent = -13;
float acc_xyz[240] = {0};
int index_acc=0;

static const char *TAG = "i2c-simple-example";

#define I2C_MASTER_SCL_IO 6      /*!< gpio number for I2C master clock */
#define I2C_MASTER_SDA_IO 7      /*!< gpio number for I2C master data  */
#define I2C_MASTER_NUM I2C_NUM_0  /*!< I2C port number for master dev */
#define I2C_MASTER_FREQ_HZ 400000 /*!< I2C master clock frequency */

static i2c_bus_handle_t i2c_bus = NULL;
static mpu6050_handle_t mpu6050 = NULL;



extern "C" void app_main(void)
{
    
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .clk_flags = 0,
    };
        conf.master.clk_speed = I2C_MASTER_FREQ_HZ;

    i2c_bus = i2c_bus_create(I2C_MASTER_NUM, &conf);
    mpu6050 = mpu6050_create(i2c_bus, MPU6050_I2C_ADDRESS);
    uint8_t mpu6050_deviceid;
    mpu6050_acce_value_t acce;
    // int cnt = 10;
    mpu6050_get_deviceid(mpu6050, &mpu6050_deviceid);
    printf("mpu6050 device ID is: 0x%02x\n", mpu6050_deviceid);
    // mpu6050_wake_up(mpu6050);
    mpu6050_set_acce_fs(mpu6050, ACCE_FS_4G);
    // mpu6050_set_gyro_fs(mpu6050, GYRO_FS_500DPS);

while(1){

for (int i=0 ;i<80; i++)
{

    mpu6050_get_acce(mpu6050, &acce);
    // printf("acce_x:%.2f, acce_y:%.2f, acce_z:%.2f\n", acce.acce_x, acce.acce_y, acce.acce_z);
    
    // acc_xyz[i]={acc_value.raw_acce_x,acc_value.raw_acce_y,acc_value.raw_acce_z};
    acc_xyz[index_acc]=acce.acce_x;
    index_acc=index_acc+1;
    acc_xyz[index_acc]=acce.acce_y;
    index_acc=index_acc+1;
    acc_xyz[index_acc]=acce.acce_z;
    index_acc=index_acc+1;
    // ESP_LOGI(TAG, "%f\n",acc_xyz[i]);
    vTaskDelay(50 / portTICK_RATE_MS);
    // printf("%d",portTICK_RATE_MS);
}
// printf("The value of j %d\n",j);
index_acc=0;

int16_t *model_input = (int16_t *)dl::tool::malloc_aligned_prefer(input_height*input_width*input_channel, sizeof(int16_t *));
    for(int i=0 ;i<input_height*input_width*input_channel; i++){
        float normalized_input = acc_xyz[i] / 1.0; //normalization
        model_input[i] = (int16_t)DL_CLIP(normalized_input * (1 << -input_exponent), -32768, 32767);
    } 

Tensor<int16_t> input;
                input.set_element((int16_t *) model_input).set_exponent(input_exponent).set_shape({input_height,input_width,input_channel}).set_auto_free(false);
                ACTIVITY model;
                dl::tool::Latency latency;
                latency.start();
                model.forward(input);
                latency.end();
                latency.print("\nActivity model", "forward");
                float *score = model.l6.get_output().get_element_ptr();
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
                    printf("0: Downstairs");
                    break;
                    case 1:
                    printf("1: Jogging");
                    break;
                    case 2:
                    printf("2: Sitting");
                    break;
                    case 3:
                    printf("3: Standing");
                    break;
                    case 4:
                    printf("4: Upstairs");
                    break;
                    case 5:
                    printf("5: Walking");
                    break;
                    default:
                    printf("No result");

                }
                printf("\n");

}
}
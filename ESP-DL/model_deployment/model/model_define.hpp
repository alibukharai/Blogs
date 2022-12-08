#pragma once
#include "dl_layer_model.hpp"
#include "dl_layer_base.hpp"
#include "dl_layer_max_pool2d.hpp"
#include "dl_layer_conv2d.hpp"
#include "dl_layer_concat.hpp"
#include "handrecognition_coefficient.hpp"
#include "dl_layer_reshape.hpp"
#include "dl_layer_softmax.hpp"
#include <stdint.h>

using namespace dl;
using namespace layer;
using namespace handrecognition_coefficient;

class HANDRECOGNITION : public Model<int16_t> // Derive the Model class in "dl_layer_model.hpp"
{
private:
    // Declare layers as member variables
    // Reshape<int16_t> l1;
    Conv2D<int16_t> l1;
    MaxPool2D<int16_t> l2;
    Conv2D<int16_t> l3;
    MaxPool2D<int16_t> l4;
    Conv2D<int16_t> l5;
    MaxPool2D<int16_t> l6;
    Reshape<int16_t> l7;
    Conv2D<int16_t> l8;
    Conv2D<int16_t> l9;
    

public:
    Softmax<int16_t> l10; //Output layer

    /**
     * @brief Initialize layers in constructor function
     * 
     */
    HANDRECOGNITION () : 
    // l1(Reshape<int16_t>({96,96,1})),
                         l1(Conv2D<int16_t>(-8, get_statefulpartitionedcall_sequential_1_conv2d_3_biasadd_filter(), get_statefulpartitionedcall_sequential_1_conv2d_3_biasadd_bias(), get_statefulpartitionedcall_sequential_1_conv2d_3_biasadd_activation(), PADDING_VALID, {}, 1,1, "l1")),
                         l2(MaxPool2D<int16_t>({2,2},PADDING_VALID, {}, 2, 2, "l2")),                      
                         l3(Conv2D<int16_t>(-9, get_statefulpartitionedcall_sequential_1_conv2d_4_biasadd_filter(), get_statefulpartitionedcall_sequential_1_conv2d_4_biasadd_bias(), get_statefulpartitionedcall_sequential_1_conv2d_4_biasadd_activation(), PADDING_VALID,{}, 1,1, "l3")),                       
                         l4(MaxPool2D<int16_t>({2,2},PADDING_VALID,{}, 2, 2, "l4")),                       
                         l5(Conv2D<int16_t>(-9, get_statefulpartitionedcall_sequential_1_conv2d_5_biasadd_filter(), get_statefulpartitionedcall_sequential_1_conv2d_5_biasadd_bias(), get_statefulpartitionedcall_sequential_1_conv2d_5_biasadd_activation(), PADDING_VALID,{}, 1,1, "l5")),                    
                         l6(MaxPool2D<int16_t>({2,2},PADDING_VALID,{}, 2, 2, "l6")),
                         l7(Reshape<int16_t>({1,1,6400},"l7_reshape")), 
                         l8(Conv2D<int16_t>(-9, get_fused_gemm_0_filter(), get_fused_gemm_0_bias(), get_fused_gemm_0_activation(), PADDING_VALID, {}, 1, 1, "l8")),
                         l9(Conv2D<int16_t>(-9, get_fused_gemm_1_filter(), get_fused_gemm_1_bias(), NULL, PADDING_VALID,{}, 1,1, "l9")),
                         l10(Softmax<int16_t>(-14,"l10")){}

                
        
    /**
     * @brief call each layers' build(...) function in sequence
     * 
     * @param input 
     */
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
        // this->l11.build(this->l10.get_output());
        
    }

    /**
     * @brief call each layers' call(...) function in sequence
     * 
     * @param input 
     */
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

        // this->l11.call(this->l10.get_output());
        // this->l10.get_output().free_element();

    }
};
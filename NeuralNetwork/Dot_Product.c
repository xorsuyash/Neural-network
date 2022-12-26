#include <stdio.h>
#define NET_INPUT_LAYER_SIZE 4 // can be replaced with (sizeof(var)/sizeof(double))
#define NET_OUTPUT_LAYER_SIZE 3
double dot_product(double *input,double *weights,double *bias,int input_size){
    int i = 0;
    double output = 0.0;
    for(i = 0;i<input_size;i++){
        output += input[i]*weights[i];
    }
    output += *bias;
    return output;
}
void layer_output(double *input,double *weights,double *bias,int input_size,double *outputs,int output_size){
    int i = 0;
    int offset = 0;
    for(i = 0; i < output_size; i++){
        outputs[i] = dot_product(input,weights + offset,&bias[i],input_size);
        offset+=input_size;
    }
}
int main(void)
{
    double input[NET_INPUT_LAYER_SIZE] = {1.0, 2.0, 3.0, 2.5};
    double weights[NET_OUTPUT_LAYER_SIZE][NET_INPUT_LAYER_SIZE] = {
                                                                {0.2, 0.8, -0.5, 1.0},
                                                                {0.5, -0.91, 0.26, -0.5},
                                                                {-0.26, -0.27, 0.17, 0.87},
                                                                };
    double bias[NET_OUTPUT_LAYER_SIZE] = {2.0,3.0,0.5};

    double output[NET_OUTPUT_LAYER_SIZE] = {0.0};
    layer_output(&input[0],&weights[0][0],&bias[0],NET_INPUT_LAYER_SIZE,&output[0],NET_OUTPUT_LAYER_SIZE);
    printf("nur output: %f %f %f\n",output[0],output[1],output[2]);

    return 0;
}
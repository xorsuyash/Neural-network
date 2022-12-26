
#include <stdio.h>
#include <stdlib.h>

#define RAND_HIGH_RANGE (0.10)
#define RAND_MIN_RANGE (-0.10)
#define INIT_BIASES (0.0)

#define NET_BATCH_SIZE 3
#define NET_INPUT_LAYER_1_SIZE 4 // can be replaced with (sizeof(var)/sizeof(double))
#define NET_HIDDEN_LAYER_2_SIZE 5 // can be replaced with (sizeof(var)/sizeof(double))
#define NET_OUTPUT_LAYER_SIZE 2 // can be replaced with (sizeof(var)/sizeof(double))


typedef struct{
    double *weights;
    double *biase;
    double *output;
    int input_size;
    int output_size;
}layer_dense_t;


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

// generate a random floating point number from min to max
double rand_range(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}



void layer_init(layer_dense_t *layer,int intput_size,int output_size){

    layer->input_size = intput_size;
    layer->output_size = output_size;

    //create data as a flat 1-D dataset
    layer->weights = malloc(sizeof(double) * intput_size * output_size);
    if(layer->weights == NULL){
        printf("weights mem error\n");
        return;
    }
    layer->biase   = malloc(sizeof(double) * output_size);
    if(layer->biase == NULL){
        printf("biase mem error\n");
        return;
    }
    layer->output = malloc(sizeof(double) * output_size);

    if(layer->output == NULL){
        printf("output mem error\n");
        return;
    }

    int i = 0;
    for(i = 0; i < (output_size); i++){
           layer->biase[i] = INIT_BIASES;
    }
    for(i = 0; i < (intput_size*output_size); i++){
           layer->weights[i] = rand_range(RAND_MIN_RANGE,RAND_HIGH_RANGE);
    }
}

//free the memory allocated by a layer
void deloc_layer(layer_dense_t *layer){
    if(layer->weights != NULL){
        free(layer->weights);
    }
    if(layer->biase != NULL){
        free(layer->biase);
    }
    if(layer->biase != NULL){
        free(layer->output);
    }
}


void forward(layer_dense_t *previos_layer,layer_dense_t *next_layer){
    layer_output((previos_layer->output),next_layer->weights,next_layer->biase,next_layer->input_size,(next_layer->output),next_layer->output_size);
}


int main()
{

    //seed the random values
    srand(0);

    int i = 0;
    int j = 0;
    layer_dense_t X;
    layer_dense_t layer1;
    layer_dense_t layer2;
    double X_input[NET_BATCH_SIZE][NET_INPUT_LAYER_1_SIZE] = {
        {1.0,2.0,3.0,2.5},
        {2.0,5.0,-1.0,2.0},
        {-1.5,2.7,3.3,-0.8}
    };


    layer_init(&layer1,NET_INPUT_LAYER_1_SIZE,NET_HIDDEN_LAYER_2_SIZE);
    layer_init(&layer2,NET_HIDDEN_LAYER_2_SIZE,NET_OUTPUT_LAYER_SIZE);

    for(i = 0; i < NET_BATCH_SIZE;i++){
        X.output = &X_input[i][0];

        forward(&X,&layer1);

     

        forward(&layer1,&layer2);
        printf("batch: %d layerY_output: ",i);
        for(j = 0; j < layer2.output_size; j++){
            printf("%f ",layer2.output[j]);
        }
        printf("\n");
    }


    deloc_layer(&layer1);
    deloc_layer(&layer2);

    return 0;
}



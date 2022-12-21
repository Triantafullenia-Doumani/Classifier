#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Hyper parameters
#define TOTAL_LAYERS 5
#define d 2 // not sure if it has to be 2
#define K 3 // Number of categories
#define H1_NEURONS 6
#define H2_NEURONS 4
#define H3_NEURONS 2
#define ACTIVATION_FUNCTION 0 //0 for "tanh", 1 for "relu"

#define LEARNING_RATE 0.15
#define EXIT_THRESHOLD 0.01

#define TRAINING_DATA 5 // TODO-> change to 4000
#define TESTING_DATA 5 // TODO-> change to 4000


/* Encoding the categories (defining the desired outputs for each category) as vectors.
e.g. 
C1 -> (1,0,0)
C2 -> (0,1,0)
C3 -> (0,0,1) 

We will use later to calculate the total error because here we will store the correct output.
*/
struct vector{
    int vec[K];
};

// Struct for each set of x1,x2,c(category)
struct data {
	float x1;
	float x2;
	int c; //category
    struct vector vector;
};

// MPL neuron
struct neuron{
    float *out_weights;
    float bias; // polwsh 
    float value; // value of neuron each moment
    float actv; // total input. Activates the neuron

    float *d_out_weights; // derivative of weights 
    float d_bias; //derivative of bias 
    float d_value; // derivative of value
    float d_actv; // derivative of act 
};

// All layers of the MultiLayer Perceptron(MPL)
struct layer{
    int neurons_num;
    struct neuron *network ; 
};

// Global 
struct data training_data[TRAINING_DATA];
struct data testing_data[TESTING_DATA];

struct layer layers[TOTAL_LAYERS];

int checkIfOutputLayer(int i){
    if(i < TOTAL_LAYERS -1){
        return 0; 
    }return 1; // Output Layer 
}
//----------------------------------CREATE ARCHITECTURE-----------------------------------------------------------------------
struct neuron initialize_neuron(int out_weights_number){
    struct neuron neuron;
    if(!(neuron.out_weights = (float*) malloc(out_weights_number * sizeof(float)))){
        printf("Initialize neuron failed");
        exit(2);
    }
    if(!(neuron.d_out_weights = (float*) malloc(out_weights_number * sizeof(float)))){
        printf("Initialize neuron failed");
        exit(2);
    }
    return neuron;
}

struct layer initialize_layer(int layer_size){
    struct layer layer;
    layer.neurons_num = layer_size;
    if(layer.network = (struct neuron*)malloc(layer_size * sizeof(struct neuron))){
        return layer;
    }
    printf("Initialize layer failed.");
    exit(3);
}

// Initialize random weights for Layers: Input + Hidden
void initialize_weights(){
    int i,k,j;
    for(i=0; i<TOTAL_LAYERS -1; i++){ 
        for(j=0; j<layers[i].neurons_num; j++){ // 2->6->4->2 (Neurons in layers 1-4)
            for(k=0; k<layers[i + 1].neurons_num; k++){ // number of output weights == number of neurons in the next layer
                layers[i].network[j].out_weights[k] = ((float)(rand() % 200) / (float)100) - 1; } } } // [-1,1]

}
// Initialize biases for Layers: Hidden + Output
// Input layer does not contain biases
void initialize_biases(){
    int i,j;
    for(i=1; i<TOTAL_LAYERS; i++){ 
        for(j=0; j<layers[i].neurons_num; j++){ // 2->6->4->2 (Neurons in layers 1-4)
            layers[i].network[j].bias = ((float)(rand() % 200) / (float)100) - 1;} }//[-1,1]
}
/* 
Initialize number of neurons per layer
Initialize weights
Initialize biases
*/
int createArchitecture(){
    int layer_size[TOTAL_LAYERS] = {d, H1_NEURONS, H2_NEURONS, H3_NEURONS, K};
    int i,j;
    for(i=0; i<TOTAL_LAYERS; i++){
        layers[i] = initialize_layer(layer_size[i]);
        for(j=0; j<layer_size[i]; j++){
            // output layer had only 1 weight, which is the final output
            if(!checkIfOutputLayer(i)){
                layers[i].network[j] = initialize_neuron(layer_size[i+1]);
            }
        }
    }
    printf("Created Layers: %d\n",TOTAL_LAYERS);
    initialize_weights();
    initialize_biases();
}
//------------------------------------------------LOAD DATASETS-----------------------------------------------
// Read from buffer(file line), and create a new data structure of type Data
struct data createDataStruct(char *buffer){
    // copy buffer into a string to remove "," and get each element(x1,x2,c)
    char *ptr = strtok(buffer, ","); 
    char *strings[30];
    unsigned char i = 0;
    while(ptr != NULL){
        strings[i] = ptr;
        ptr = strtok(NULL, ",");
        i++;
    }
    struct data data;
    data.x1 = atof(strings[0]);
    data.x2 = atof(strings[1]);
    data.c = atof(strings[2]);
    for(i=0; i<K; i++){
        // e.g if 3 = C3 then (0,0,1)
        if(i == data.c - 1){
            data.vector.vec[i]=1;
        }else{
            data.vector.vec[i]=0;
        }
    }
    return data;

}
void loadDataset(char *filename,struct data *dataset, int dataset_size){
    char buffer[30];
    FILE *fptr;
    struct data data;
    int i,k;
    if ((fptr = fopen(filename,"r")) == NULL){
        printf("Error! opening file");
        // Program exits if the file pointer returns NULL.
        exit(0);
    }
    // reading line by line
    i = 0;
    while (fgets(buffer, 30, fptr) != NULL){
        data = createDataStruct(buffer);
        if(i > dataset_size){
            printf("'%s': File size is bigger than expected:(%d)",filename,dataset_size);
            exit(1);
        }
        dataset[i].x1 = data.x1;
        dataset[i].x2 = data.x2;
        dataset[i].c = data.c;
        for(k=0; k<K; k++){
            dataset[i].vector.vec[k] = data.vector.vec[k];
        }
        i++;
   }
    fclose(fptr);
}
//----------------------------------FORWARD PASS------------------------------------
void putInput(float x1, float x2){
    layers[0].network[0].actv =  x1;
    layers[0].network[1].actv =  x2;
}
void relu(int layer, int neuron){
    if(layers[layer].network[neuron].value < 0){
        layers[layer].network[neuron].actv = 0;
    }else{
        layers[layer].network[neuron].actv = layers[layer].network[neuron].value;
    }
}
void tahn(int layer, int neuron){
    float x = layers[layer].network[neuron].value;
    float tahn_result =  ((float)exp(x) - (float)exp(-x)) / (float)(exp(x) + (float)exp(-x));
    layers[layer].network[neuron].actv = tahn_result;
}
void sigmoid(int layer, int neuron){
    float x = layers[layer].network[neuron].value;
    float sigmoid_result = (float)1.0/((float)1.0+(float)exp(-(x)));
    layers[layer].network[neuron].actv = sigmoid_result;
}
void activationFunction(int layer, int neuron){
    if(!checkIfOutputLayer(layer)){ // Hidded Layer
        if(ACTIVATION_FUNCTION == 0 ){ 
            tahn(layer,neuron);
        }
        else{                         
            relu(layer,neuron);
        }
    }else{                         // Output Layer
        sigmoid(layer,neuron);
    }
}

void forwardPass(){
    int i,j,k;
    float prev_out_weight;
    float prev_actv;
    int act_fun;
    int sum = 0;
    for(i=1; i<TOTAL_LAYERS; i++){
        for(j=0; j<layers[i].neurons_num; j++){
            layers[i].network[j].value = layers[i].network[j].bias;

            for(k=0; k<layers[i-1].neurons_num; k++){
                prev_actv       = layers[i-1].network[k].actv;
                prev_out_weight = layers[i-1].network[k].out_weights[j];

                layers[i].network[j].value +=  prev_actv * prev_out_weight;
                // Now we need to use the activation function to update the 'actv' value
                activationFunction(i,j);
            }
        }
    }
} 

//----------------------------------BACK PROPAGATION--------------------------------
// back propagation for Output Layer
void backPropagationOutputLayer(struct vector vector){
    int i,k;
    float actv,d_value;
    float prev_actv, prev_weight; // act and weight of a neuron of the previous layer
    float correct_out;

    for(i=0; i<K; i++){
        prev_actv = layers[TOTAL_LAYERS - 1].network[i].actv;
        correct_out = vector.vec[i];
        layers[TOTAL_LAYERS -1].network[i].d_value = actv - correct_out*actv*(1 - actv);

        for(k=0; k<H3_NEURONS; k++){
            prev_actv = layers[TOTAL_LAYERS-2].network[k].actv;
            d_value = layers[TOTAL_LAYERS-1].network[i].d_value;
            layers[TOTAL_LAYERS-2].network[k].d_out_weights[i] = prev_actv * d_value;
            
            prev_weight = layers[TOTAL_LAYERS-2].network[k].out_weights[i];
            layers[TOTAL_LAYERS-2].network[k].d_actv = prev_weight * d_value;
        }
        layers[TOTAL_LAYERS-1].network[i].d_bias = d_value;
    }
}
void backPropagationHiddenLayers(){
    int i,j,k;
    for(i=TOTAL_LAYERS -2; i==1; i--){
        for(j=0;j<layers[i].neurons_num; j++){
            if(layers[i].network[j].value >=0){
                layers[i].network[j].d_value = layers[i].network[j].d_actv;
            }else{
                layers[i].network[j].d_value = 0;
            }
            for(k=0; k<layers[i-1].neurons_num; k++){
                layers[i-1].network[k].d_out_weights[j] = layers[i].network[j].d_value * layers[i-1].network[k].actv;
                if(i>1){
                    layers[i-1].network[k].d_actv = layers[i-1].network[k].out_weights[j] * layers[i].network[j].d_value;
                }
            }
            layers[i].network[j].d_bias = layers[i].network[j].d_value;
        }
    }
}
void backPropagation(struct vector vector){
    backPropagationOutputLayer(vector);
    backPropagationOutputLayer();
}

//----------------------------------PRINT-------------------------------------------
void printDataset(char *dataset_name, struct data *dataset, int dataset_size){
    int i,k;
    printf("%s\n",dataset_name);
    for(i=0; i < dataset_size; i++){
        printf("x1:%f, x2:%f, c:%d, vector:( ",dataset[i].x1,dataset[i].x2,dataset[i].c);
        for(k=0; k<K; k++){
            printf("%d ",dataset[i].vector.vec[k]);
        }
        printf(")\n");
    }
}

void printLayers(){
    int i,j,k;
    int out_weights_size;
    for(i=0; i<TOTAL_LAYERS; i++){
        printf("\nLAYER: %d\nNEURONS: %d\n", i, layers[i].neurons_num);
        for(j=0; j<layers[i].neurons_num; j++){
            
            printf("(neuron %d) ",j);
            printf("BIAS: %f ",layers[i].network[j].bias);
            if( i < TOTAL_LAYERS -1){    
                out_weights_size = sizeof(layers[i].network[j].out_weights)/sizeof(layers[i].network[j].out_weights[0]);
                printf("WEIGHTS: ");
                for(k=0; k<out_weights_size; k++){
                    printf("%f ",layers[i].network[j].out_weights[k]);
                }
            }printf("\n");
        }
    }
}
//-----------------------------------------ERROR-------------------------------------
float calculateError(struct vector vector){
    float diff,error = 0;
    int i;
    for(i=0; i<K; i++){ // K = 3(num of categories)
        diff = vector.vec[i] - layers[TOTAL_LAYERS -1].network[i].actv;
        error += (float)pow(diff,2);
    }
    error = ((float)0.5*error);
    return error;
}
//----------------------------------------TRAIN-------------------------------------
void trainNetwork(){
    int i,j;
    float total_error = 0;
    for(i=0; i<TRAINING_DATA; i++){
        putInput(training_data[i].x1, training_data[i].x2);
        forwardPass();
        total_error+= calculateError(training_data->vector);
        backPropagation(training_data->vector);
    }
    printf("Total Error after training: %f",total_error);
}
//----------------------------------------MAIN--------------------------------------
void main(){
    loadDataset("training_data.txt", training_data, TRAINING_DATA);
    loadDataset("testing_data.txt", testing_data, TESTING_DATA);
    createArchitecture();
    trainNetwork();
    printLayers();
    printDataset("Training data",training_data,TRAINING_DATA);
    //printDataset("Testing data",testing_data,TESTING_DATA);
    //createTestingDataset()
    //createTrainingDataset
}

/*
1) Why Initialize a Neural Network with Random Weights? https://machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/

2) How does the bias works in Neural Networks? 
Bias can be considered as an additional set of weights in a model that doesn’t need any input, and related to the output of the model when it has no inputs.
Adding a constant to the input bias allows to shift the activation function. 
There, a bias works exactly the same way it does in a linear equation:

bias = mx + c

In a scenario with bias, the input to the activation function is 'x' times the connection weight 'w0' plus the bias times the connection weight for the bias 'w1'. This has the effect of shifting the activation function by a constant amount (b * w1).

3)forward-pass https://www.google.com/search?q=forward-pass+in+neural+network&source=lnms&tbm=vid&sa=X&ved=2ahUKEwiRgMb2lon8AhUoS_EDHd4bCJoQ_AUoAnoECAEQBA&biw=768&bih=702&dpr=1.25#fpstate=ive&vld=cid:ea47a04d,vid:UJwK6jAStmg
*/
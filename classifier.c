#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define d 2 
#define K 3 // Number of categories
#define H1_NEURONS 6
#define H2_NEURONS 4
#define H3_NEURONS 2
#define ACTIVATION_FUNCTION 0 //0 for "tanh", 1 for "relu"

#define TOTAL_LAYERS 5

#define LEARNING_RATE ?
#define EXIT_THRESHOLD ?

#define DATASET_SIZE  5

/* Encoding the categories (defining the desired outputs for each category) as vectors.
e.g. 
C1 -> (1,0,0)
C2 -> (0,1,0)
C3 -> (0,0,1)  
*/

//stackoverflow 241 likes: ' PLEASE don't typedef structs in C, it needlessly pollutes the global namespace which is typically very polluted already in large C programs.
struct vector{
    int vec[K];
};

// Struct for each set of x1,x2,c(category)
struct data {
	float x1;
	float x2;
	int c;
    struct vector vector;
};

// MPL neuron
struct neuron{
    float *out_weights;
    float bias;
};

struct layer{
    int neurons_num;
    struct neuron *network ; 
};

// Global 
struct data training_data[DATASET_SIZE];
struct data testing_data[DATASET_SIZE];

struct layer layers[TOTAL_LAYERS];

// Initialize the number of neurons per layer, and initialize the weights
int create_architecture(){
    int layer_size[TOTAL_LAYERS] = {d, H1_NEURONS, H2_NEURONS, H3_NEURONS, K};
    int i,j;
    for(i=0; i<TOTAL_LAYERS; i++){
        layers[i] = initialize_layer(layer_size[i]);

        for(j=0; j<layer_size[i]; j++){
            if(i < TOTAL_LAYERS - 1){
                layers[i].network[j] = initialize_neuron(layer_size[i+1]);
            }
        }
    }
    initialize_weights();
}

struct neuron initialize_neuron(int out_weights_number){
    struct neuron neuron;
    if(neuron.out_weights = (float*) malloc(out_weights_number * sizeof(float))){
        return neuron;
    }
    printf("Initialize neuron failed.");
    exit(2);
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

void initialize_weights(){
    int i,k,j;
    for(i=0; i<TOTAL_LAYERS -1; i++){
        for(j=0; j<layers[i].neurons_num; j++){
            for(k=0; k<layers[i + 1].neurons_num; k++){
                layers[i].network[j].out_weights[k] = (float)rand() / (float)RAND_MAX ; // [0,1]
                print("weight: %f", layers[i].network[j].out_weights[k]);
            }
            // I don't get that
            if(i > 0){
                layers[i].network[j].bias = (float)rand() / (float)RAND_MAX ; //[0,1]
            }
        }
    }// don't get it
    for(j=0; j < layers[TOTAL_LAYERS-1].neurons_num; j++){
        layers[TOTAL_LAYERS-1].network[j].bias = (float)rand() / (float)RAND_MAX ; //[0,1]
    }
}
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
void loadDataset(char *filename,struct data *dataset){
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
        if(i > DATASET_SIZE){
            printf("'%s': File size is bigger than expected:(%d)",filename,DATASET_SIZE);
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

void printDataset(char *dataset_name, struct data *dataset){
    int i,k;
    printf("%s\n",dataset_name);
    for(i=0; i < DATASET_SIZE; i++){
        printf("x1:%f, x2:%f, c:%d, vector:( ",dataset[i].x1,dataset[i].x2,dataset[i].c);
        for(k=0; k<K; k++){
            printf("%d ",dataset[i].vector.vec[k]);
        }
        printf(")\n");
    }
}
void main(){
    loadDataset("training_data.txt", training_data);
    loadDataset("testing_data.txt", testing_data);
    printDataset("Training data",training_data);
    printDataset("Testing data",testing_data);
    //createTestingDataset()
    //createTrainingDataset
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define d 2 
#define K 3 // Number of categories
#define H1_NEURONS 6
#define H2_NEURONS 4
#define H3_NEURONS 2
#define ACTIVATION_FUNCTION "tanh" // or "relu"

#define DATASET_SIZE  5

/* Encoding the categories (defining the desired outputs for each category) as vectors.
e.g. 
C1 -> (1,0,0)
C2 -> (0,1,0)
C3 -> (0,0,1)  
*/
typedef struct vector{
    int vec[K];
}Vector;

// Struct for each set of x1,x2,c(category)
typedef struct data {
	float x1;
	float x2;
	int c;
    Vector vector;
}Data;

Data training_data[DATASET_SIZE];
Data testing_data[DATASET_SIZE];

// Read from buffer(file line), and create a new data structure of type Data
Data createDataStruct(char *buffer){
    // copy buffer into a string to remove "," and get each element(x1,x2,c)
    char *ptr = strtok(buffer, ","); 
    char *strings[30];
    unsigned char i = 0;
    while(ptr != NULL){
        strings[i] = ptr;
        ptr = strtok(NULL, ",");
        i++;
    }
    Data data;
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
void loadDataset(char *filename,Data *dataset){
    char buffer[30];
    FILE *fptr;
    Data data;
    int i,k;
    if ((fptr = fopen(filename,"r")) == NULL){
        printf("Error! opening file");
        // Program exits if the file pointer returns NULL.
        exit(1);
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

void printDataset(char *dataset_name, Data *dataset){
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
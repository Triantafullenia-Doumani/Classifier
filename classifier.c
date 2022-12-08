#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define d 2
#define K 3 
#define H1_NEURONS 6
#define H2_NEURONS 4
#define H3_NEURONS 2
#define ACTIVATION_FUNCTION "tanh" // or "relu"

// Struct for each set of x1,x2,c(category)
typedef struct data {
	float x1;
	float x2;
	int c;
}Data;

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
    /*
    printf("x1: %f\n",data.x1);
    printf("x2: %f\n",data.x2);
    printf("c: %d\n",data.c);*/
    return data;

}
void loadDataset(char *filename){
    char buffer[30];
    FILE *fptr;
    if ((fptr = fopen(filename,"r")) == NULL){
        printf("Error! opening file");
        // Program exits if the file pointer returns NULL.
        exit(1);
    }
    // reading line by line
    while (fgets(buffer, 30, fptr) != NULL){
        createDataStruct(buffer);
   }
    fclose(fptr);
}
void main(){
    loadDataset("training_data.txt");
    loadDataset("testing_data.txt");
    //createTestingDataset()
    //createTrainingDataset
}
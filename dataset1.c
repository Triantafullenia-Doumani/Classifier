#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>


void addData(FILE *fptr,float x1,float x2, int c){
	fprintf(fptr,"%.4f,%.4f,%d\n",x1,x2,c);
}

void setToCategory(FILE *fptr,float x1,float x2){
	int c = 3;
	// 1) εάν (x1 – 0.5)2 + (x2 – 0.5)2 <0.2, και x2>0.5 τότε το (x1,x2) κατατάσσεται στην κατηγορία C1
	if (((pow((x1 - 0.5),2) + pow((x2 - 0.5),2)) < 0.2) && (x2 > 0.5)){
		c = 1;
	}
	// 2) εάν (x1 – 0.5)2 + (x2 – 0.5)2 <0.2, και x2<0.5 τότε το (x1,x2) κατατάσσεται στην κατηγορία C2
	else if(((pow((x1 - 0.5),2) + pow((x2 - 0.5),2)) < 0.2) && (x2 < 0.5)){
		c = 2;
	}
	// 3) εάν (x1 + 0.5)2 + (x2 + 0.5)2 <0.2, και x2>-0.5 τότε το (x1,x2) κατατάσσεται στην κατηγορία C1,
	else if(((pow((x1 + 0.5),2) + pow((x2 + 0.5),2)) < 0.2) && (x2 > -0.5)){
		c = 1;
	}
	// 4) εάν (x1 + 0.5)2 + (x2 + 0.5)2 <0.2, και x2<-0.5 τότε το (x1,x2) κατατάσσεται στην κατηγορία C2
	else if(((pow((x1 + 0.5),2) + pow((x2 + 0.5),2)) < 0.2) && (x2 < -0.5)){
		c = 2;
	}
	// 5) εάν (x1 – 0.5)2 + (x2 + 0.5)2 <0.2, και x2>-0.5 τότε το (x1,x2) κατατάσσεται στην κατηγορία C1
	else if(((pow((x1 - 0.5),2) + pow((x2 + 0.5),2)) < 0.2) && (x2 > -0.5)){
		c = 1;
	}
	// 6) εάν (x1 – 0.5)2 + (x2 + 0.5)2 <0.2, και x2<-0.5 τότε το (x1,x2) κατατάσσεται στην κατηγορία C2
	else if(((pow((x1 - 0.5),2) + pow((x2 + 0.5),2)) < 0.2) && (x2 < -0.5)){
		c = 2;
	}
	// 7) εάν (x1 + 0.5)2 + (x2 - 0.5)2 <0.2, και x2>0.5 τότε το (x1,x2) κατατάσσεται στην κατηγορία C1
	else if(((pow((x1 + 0.5),2) + pow((x2 - 0.5),2)) < 0.2) && (x2 > 0.5)){
		c = 1;
	}
	// 8) εάν (x1 + 0.5)2 + (x2 - 0.5)2 <0.2, και x2<0.5 τότε το (x1,x2) κατατάσσεται στην κατηγορία C2
	else if(((pow((x1 + 0.5),2) + pow((x2 - 0.5),2)) < 0.2) && (x2 < 0.5)){
		c = 2;
	}
	addData(fptr,x1,x2,c);

}
//Generate random numbers within a specific range
//argument lower-> value of the minimum number
//argument upper-> value of the maximum number
float generateRandomNumbers(float lower,float upper)
{
	return ((float)rand()*(upper-lower)) / (float)RAND_MAX+lower;
}

//Fill the dataset with random values in a specific range
void generateRandomPoints(FILE *fptr)
{
	srand(time(0));
	for(int i=0; i<4000; i++)
	{	
		float x1 = generateRandomNumbers(-1.1,1.0);
		float x2 = generateRandomNumbers(-1.1,1.0);
		//fprintf(fptr,"%.4f,%.4f\n",x1,x2);
		setToCategory(fptr,x1,x2);
	}
}

void createDataset(char *dataset_name){
	FILE *fptr;
	fptr = fopen(dataset_name,"w");

	if(fptr==NULL)
	{
		printf("ERROR Creating File!");
		exit(1);
	}

	generateRandomPoints(fptr);
	fclose(fptr);
}

int main(void)
{
	createDataset("training_data.txt");
	createDataset("testing_data.txt");
	return 0;
}
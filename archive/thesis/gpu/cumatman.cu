#include "cumatman.h"

/**
* Allocate array of float initialized to all bits 0.
* Returns 1 if there is an error, 0 otherwise
*/
int allocate_array(float **a, int m, int n) {
	cudaMallocHost((void **)a, m*n*sizeof(float));
	// if( (*a = (float *)calloc(m*n, sizeof(float))) == NULL )
	// return 1;
	return 0;
}

/**
* Allocate array of int initialized to all bits 0.
* Returns 1 if there is an error, 0 otherwise
*/
int allocate_int_array(int **a, int m, int n) {
	cudaMallocHost((void **)a, m*n*sizeof(int));
	// if( (*a = (int *) calloc(m * n, sizeof( int ))) == NULL )
	// return 1;
	return 0;
}

// Print an array of float in the proper format
void display_array(const char *name, float *a, int m, int n) {
	int i, j;
	printf("Array %s:\n", name);
	for(i=0;i<m;i++) {
		for(j=0; j<n;j++)
			printf("%f ", a[R2C(i,j,m)]);
		printf("\n");
	}
}

//Print an array of integer in the proper format
void display_int_array(const char *name, int *a, int m, int n) {
	int i, j;
	printf("Int array %s:\n", name);
	for(i=0;i<m;i++) {
		for(j=0; j<n;j++)
			printf("%d ", a[R2C(i,j,m)]);
		printf("\n");
	}
}

/**
* Read array from standard input.
*/
int read_array(FILE *file, float *a, int m, int n) {
	int i,j;
	//Data from the standard input.
	for(i=0; i<m; i++)
		for(j=0; j<n; j++) {
			fscanf(file, "%f", &a[R2C(i,j,m)]); //Get the ith-element of the matrix from
		 } //the command line, converting it
	//from text to float
	return 0;
}

// Release allocated memory
void free_array(void *a) {
	cudaFreeHost(a);
}
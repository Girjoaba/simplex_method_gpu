#include "matman.h"

/**
* Allocate array of float initialized to all bits 0.
* Returns 1 if there is an error, 0 otherwise
*/
int allocate_array(float **a, int m, int n) {
	if((*a = (float*)calloc(m*n, sizeof(float))) == NULL)
		return 1;
	return 0;
}

/**
* Allocate array of int initialized to all bits 0.
* Returns 1 if there is an error, 0 otherwise
*/
int allocate_int_array(int **a, int m, int n) {
	if((*a = (int*)calloc(m*n, sizeof(int))) == NULL)
		return 1;
	return 0;
}

// Print an array of float in the proper format
void display_array(char *name, float *a, int m, int n) {
	int i, j;
	printf("Array %s:\n", name);
	for(i=0; i<m; i++) {
		for(j=0; j<n; j++)
			printf("%f ", a[i*n+j]);
		printf("\n");
	}
}

//Print an array of integer in the proper format
void display_int_array(char *name, int *a, int m, int n) {
	int i, j;
	printf("Int array %s:\n", name);
	for(i=0; i<m; i++) {
		for(j=0; j<n; j++)
			printf("%d ", a[i*n+j]);
		printf("\n");
	}
}

// Read array from standard input.
int read_array(FILE *file, float *a, int m, int n) {
	int i, dim;
	dim = m*n;
	for(i = 0; i < dim; i++)
		fscanf(file, "%f", &a[i]);
	return 0;
}

// Release allocated memory
void free_array(void *a) {
	free(a);
}
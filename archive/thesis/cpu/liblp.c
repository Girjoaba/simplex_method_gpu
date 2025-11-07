#include "liblp.h"
#include "matman.h"

int entering_index(float *v, int size) {
	int i;
	int min_i = 0;
	for (i = 1; i < size; i++)
		if (v[i] < v[min_i])
			min_i = i;
	return min_i;
}

int leaving_index(float *t, int *flag, int size) {
	int i;
	int minpos_i = -1;
	for (i = 0; i < size; i++) {
		if (minpos_i < 0) {
			if ((flag[i] > 0) && (t[i] >= -EPS))
				minpos_i = i;
		} else if ((flag[i] > 0) && (t[i] >= -EPS) && (t[i] < t[minpos_i]))
			minpos_i = i;
	}
	return minpos_i;
}

void compute_theta(float *x, float *a, float *t, int *flag, int size) {
	int i;
	for (i = 0; i < size; i++)
		if (a[i] > 0) {
			flag[i] = 1;
			t[i] = x[i] / a[i];
		} else flag[i] = 0;
}

int compute_E(float *E, float *a, float *I, int size, int li) {
	int i;
	float qth = a[li];

	if ((qth >= -EPS) && (qth <= EPS)) {
		printf("qth == 0....exit...\n");
		return 1;
	}

	memcpy(E, I, size * size * sizeof(float));

	for (i = 0; i < size; i++)
		a[i] = -a[i] / qth;

	a[li] = 1 / qth;

	for (i = 0; i < size; i++)
		E[(i * size) + li] = a[i];

	return 0;
}

void extract_column(float *M, float *v, int start_i, int stride, int size) {
	int i;
	for (i = 0; i < size; i++)
		v[i] = M[start_i + (i * stride)];
}

void create_identity_matrix(float **m, int size) {
	int i;
	allocate_array(m, size, size);
	for (i = 0; i < size; i++)
		(*m)[i * size + i] = 1;
}
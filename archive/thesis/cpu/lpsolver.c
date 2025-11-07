#include "matman.h"
#include "liblp.h"
#include <sys/types.h>
#include <time.h>
#include <cblas.h>
#define MAX_ITER 1000

float *c, *A, *b;
// Binv: Basis matrix inverse
// newBinv: temporary matrix inverse for swap purposes
// E: used to compute the inversion using just one mm multiplication
// newBinv = E * Binv
float *Binv, *newBinv, *E;
// e: cost contributions vector used to determine the entering variable
// D, y, yb: arrays used to compute the cost contributions vector
// D = [-c ; A] y = cb * Binv yb = [1 y] e = yb * D
float *D, *y, *yb, *e;
float *I; // Identity matrix Im
// xb: current basis

// cb: basis costs
// xb = Binv * b
float *cb, *xb;
// A_e: entering variable column of constraint factors
// alpha: the pivotal vector used to determine the leaving variable
// theta: Increases vector
// alpha = Binv * A_e
float *A_e, *alpha, *theta;
// Vector of flags indicating valid increases
// (valid alpha[i]) <==> (theta_flag[i] == 1)
int *theta_flag;
// Vector containing basis variables’ indexes
int *bi;
// Constraints matrix dimensions m and n.
// Indexes of the entering and leaving variables.
int m, n, ev, lv;
// Cost to optimize
// z = c * x
float z;
void help();

/****************** MAIN *********************/
int main(int argc, char **argv) {
	int i, opt, ret;
	FILE *sourcefile;
	struct timespec start, end, ev_start, ev_end, lv_start, lv_end,
	b_start, b_end;
	struct timespec blas_end;
	long nsec;

	switch(argc) {
	case 2:
		if(strcmp(argv[1],"-h")==0 || strcmp(argv[1],"--help")==0) {
			help();
		} else if((sourcefile = fopen(argv[1], "r")) == NULL) {
			printf("Error opening %s\n", argv[2]);
			return 1;
		}
		break;
	default:
		printf("Wrong parameter sequence.\n");
		return 1;
	}

	clock_gettime(CLOCK_REALTIME, &start);
	fscanf(sourcefile, "%d%d", &m, &n);
	if (m > n) {
		printf("Error: it should be n>=m\n");
		return 1;
	}
	printf("m=%d n=%d\n", m, n);
	printf("Size: %d\n", m * n);

	// c
	allocate_array(&c, 1, n);
	read_array(sourcefile, c, 1, n);
	// b
	allocate_array(&b, m, 1);
	read_array(sourcefile, b, m, 1);
	// A
	allocate_array(&A, m, n);
	read_array(sourcefile, A, m, n);
	fclose(sourcefile);

	// Im
	create_identity_matrix(&I, m);
	// Binv, newBinv, E
	allocate_array(&Binv, m, m);
	allocate_array(&newBinv, m, m);
	allocate_array(&E, m, m);
	// Initialize Binv = Im
	memcpy(Binv, I, m * m * sizeof(float));
	// D, y, yb, e
	allocate_array(&D, m + 1, n);
	allocate_array(&y, 1, m);
	allocate_array(&yb, 1, m + 1);
	allocate_array(&e, 1, n);
	// Set first element of yb = 1
	yb[0] = 1;
	// Initialize D = [-c ; A]
	memcpy(D, c, n * sizeof(float));
	cblas_sscal(n, -1, D, 1);
	memcpy(&D[n], A, m * n * sizeof(float));
	// cb, xb
	allocate_array(&cb, 1, m);
	allocate_array(&xb, m, 1);
	// Initialize with the last m elements of c
	memcpy(cb, &c[n - m], m * sizeof(float));
	memcpy(xb, b, m * sizeof(float));
	// A_e, alpha, theta
	allocate_array(&A_e, m, 1);
	allocate_array(&alpha, m, 1);
	allocate_array(&theta, 1, m);
	// theta_flag & bi
	allocate_int_array(&theta_flag, 1, m);
	allocate_int_array(&bi, 1, m);
	// Initialize with the basis indexes
	for (i = 0; i < m; i++)
		bi[i] = (n - m) + i;

	// Optimization loop
	i = 0;
	do {
		clock_gettime(CLOCK_REALTIME, &ev_start);
		// y = cb*Binv
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, m, m, 1, cb, m, Binv, m, 0, y, m);
		memcpy(&yb[1], y, m * sizeof(float));
		// e = [1 y]*[-c ; A]
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, n, m + 1, 1, yb, n, D, n, 0, e, n);
		clock_gettime(CLOCK_REALTIME, &blas_end);
		ev = entering_index(e, n);
		clock_gettime(CLOCK_REALTIME, &ev_end);
		if (e[ev] >= -EPS) {
			opt = 1;
			break;
		}

		clock_gettime(CLOCK_REALTIME, &lv_start);
		extract_column(A, A_e, ev, n, m);
		// alpha = Binv*A_e
		cblas_sgemv(CblasRowMajor, CblasNoTrans, m, m, 1, Binv, m, A_e, 1, 0, alpha, 1);
		compute_theta(xb, alpha, theta, theta_flag, m);
		lv = leaving_index(theta, theta_flag, m);
		clock_gettime(CLOCK_REALTIME, &lv_end);
		if (lv < 0) {
			opt = 2;
			break;
		}

		clock_gettime(CLOCK_REALTIME, &b_start);
		if (compute_E(E, alpha, I, m, lv)) {
			opt = 3;
			break;
		}
		// Binv = E*Binv
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1, E, m, Binv, m, 0, newBinv, m);
		memcpy(Binv, newBinv, m * m * sizeof(float));
		clock_gettime(CLOCK_REALTIME, &b_end);
		
		// Update cb
		bi[lv] = ev;
		cb[lv] = c[ev];
		// xb=Binv*b
		cblas_sgemv(CblasRowMajor, CblasNoTrans, m, m, 1, Binv, m, b, 1, 0, xb, 1);
		i++;
	} while (i < MAX_ITER);

	if (opt == 1) {
		cblas_sgemv(CblasRowMajor, CblasNoTrans, 1, m, 1, cb, m, xb, 1, 0, &z, 1);
		printf("Optimum found: %f\n", z);
		for (i = 0; i < m; i++)
			printf("x_%d = %f\n", bi[i], xb[i]);
	} else if (opt == 2)
		printf("Problem unbounded.\n");
	else
		printf("Problem unsolvable: either qth==0 or loop too long.\n");

	// Deallocate arrays
	free_array(A);
	free_array(b);
	free_array(c);
	free_array(D);
	free_array(E);
	free_array(I);
	free_array(A_e);
	free_array(Binv);
	free_array(newBinv);
	free_array(cb);
	free_array(xb);
	free_array(y);
	free_array(yb);
	free_array(e);
	free_array(alpha);
	free_array(theta);
	free_array(theta_flag);
	free_array(bi);
	clock_gettime(CLOCK_REALTIME, &end);
	// Overall time
	nsec = end.tv_nsec - start.tv_nsec;
	if (nsec < 0) {
		nsec = 1E9 + nsec;
		end.tv_sec -= 1;
	}
	printf("Elapsed time: %.9f\n", (double)(end.tv_sec - start.tv_sec) + (double)(nsec * 1E-9));
	// BLAS entering variable computation time
	nsec = blas_end.tv_nsec - ev_start.tv_nsec;
	if (nsec < 0) {
		nsec = 1E9 + nsec;
		blas_end.tv_sec -= 1;
	}
	printf("BLAS entering variable computation time: %.9f\n", (double)(blas_end.tv_sec - ev_start.tv_sec) + (double)(nsec * 1E-9));
	// Entering variable computation time
	nsec = ev_end.tv_nsec - ev_start.tv_nsec;
	if (nsec < 0) {
		nsec = 1E9 + nsec;
		ev_end.tv_sec -= 1;
	}
	printf("Entering variable computation time: %.9f\n", (double)(ev_end.tv_sec - ev_start.tv_sec) + (double)(nsec * 1E-9));
	// Leaving variable computation time
	nsec = lv_end.tv_nsec - lv_start.tv_nsec;
	if (nsec < 0) {
		nsec = 1E9 + nsec;
		lv_end.tv_sec -= 1;
	}
	printf("Leaving variable computation time: %.9f\n", (double)(lv_end.tv_sec - lv_start.tv_sec) + (double)(nsec * 1E-9));
	// Binv updating time
	nsec = b_end.tv_nsec - b_start.tv_nsec;
	if (nsec < 0) {
		nsec = 1E9 + nsec;
		b_end.tv_sec -= 1;
	}

	printf("Binv updating time: %.9f\n", (double)(b_end.tv_sec - b_start.tv_sec) + (double)(nsec * 1E-9));
	return 0;
}

void help() {
	printf("Input format:\n");
	printf("M N <vector c> <vector b> <matrix A>\n");
}
#include "cumatman.h"
#include "culiblp.h"
#include <sys/types.h>
#include <time.h>

#include <cstdio>
#include <cstring>

/**
* Arrays’ indexes follow the C convention (0 <= i < N)
**/

// Main problem arrays: costs and constrains
float *c, *A, *b, *xb;
int *bi;
float z;
struct timespec ev_start, ev_end, lv_start, lv_end, b_start,
b_end, alloc_start, alloc_end, dealloc_start, dealloc_end, init_start, init_end;
struct timespec blas_end;
void help();
/****************** MAIN *********************/

int main(int argc, char **argv) {
	int i, m, n;
	FILE *sourcefile;
	struct timespec start, end, read_start, read_end, hostall_start, hostall_end;
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
	// read m and n
	fscanf(sourcefile, "%d%d", &m, &n);
	if(m>n) {
		printf("Error: it should be n>=m\n");
		return 1;
	}
	printf("m=%d n=%d\n", m, n);
	printf("Size: %d\n", m*n);

	//Initialize all arrays
	clock_gettime(CLOCK_REALTIME, &hostall_start);
	allocate_array(&c, 1, n);
	allocate_array(&b, m, 1);
	allocate_array(&A, m, n);
	clock_gettime(CLOCK_REALTIME, &hostall_end);
	clock_gettime(CLOCK_REALTIME, &read_start);
	// c
	read_array(sourcefile, c, 1, n);
	// b
	read_array(sourcefile, b, m, 1);
	// A
	read_array(sourcefile, A, m, n);
	clock_gettime(CLOCK_REALTIME, &read_end);
	//Close source file
	fclose(sourcefile);
	// xb
	allocate_array(&xb, 1, m);
	// bi
	allocate_int_array(&bi, 1, m);
	z = lpsolve(A, b, c, xb, bi, m, n);
	if(isnan(z))
		printf("Problem unsolvable: either qth==0 or loop too long.\n");
	else if(isinf(z))
		printf("Problem unbounded.\n");
	else {
		printf("Optimum found: %f\n", z);
		for(i=0; i<m; i++)
			printf("x_%d = %f\n", bi[i], xb[i]);
	}

	// Deallocate arrays
	free_array(A);
	free_array(b);
	free_array(c);
	free_array(xb);
	free_array(bi);
	clock_gettime(CLOCK_REALTIME, &end);

	nsec = end.tv_nsec-start.tv_nsec;
	if(nsec < 0) {
		nsec = 1E9+nsec;
		end.tv_sec -= 1;
	}
	printf("Elapsed time: %.9f\n", (double)(end.tv_sec-start.tv_sec)+(double)(nsec*1E-9));
	//Read computation time
	nsec = read_end.tv_nsec-read_start.tv_nsec;
	if(nsec < 0) {
		nsec = 1E9+nsec;
		read_end.tv_sec-=1;
	}
	printf("Read time: %.9f\n", (double)(read_end.tv_sec-read_start.tv_sec)+(double)(nsec*1E-9));
	//Host alloc computation time
	nsec = hostall_end.tv_nsec-hostall_start.tv_nsec;
	if(nsec < 0) {
		nsec = 1E9+nsec;
		hostall_end.tv_sec-=1;
	}
	printf("Host allocation time: %.9f\n", (double)(hostall_end.tv_sec-hostall_start.tv_sec)+(double)(nsec*1E-9));
	//BLAS entering variable computation time
	nsec = blas_end.tv_nsec-ev_start.tv_nsec;
	if(nsec < 0) {
		nsec = 1E9+nsec;
		blas_end.tv_sec-=1;
	}
	printf("BLAS entering variable computation time: %.9f\n", (double)(blas_end.tv_sec-ev_start.tv_sec)+(double)(nsec*1E-9));
	//Entering variable computation time
	nsec = ev_end.tv_nsec-ev_start.tv_nsec;
	if(nsec < 0) {
		nsec = 1E9+nsec;
		ev_end.tv_sec-=1;
	}
	printf("Entering variable computation time: %.9f\n", (double)(ev_end.tv_sec-ev_start.tv_sec)+(double)(nsec*1E-9));
	//Alloc computation time
	nsec = alloc_end.tv_nsec-alloc_start.tv_nsec;
	if(nsec < 0) {
		nsec = 1E9+nsec;
		alloc_end.tv_sec-=1;
	}
	printf("Alloc time: %.9f\n", (double)(alloc_end.tv_sec-alloc_start.tv_sec)+(double)(nsec*1E-9));
	//Dealloc computation time
	nsec = dealloc_end.tv_nsec-dealloc_start.tv_nsec;
	if(nsec < 0) {
		nsec = 1E9+nsec;
		dealloc_end.tv_sec-=1;
	}
	printf("Dealloc time: %.9f\n", (double)(dealloc_end.tv_sec-dealloc_start.tv_sec)+(double)(nsec*1E-9));
	//Init computation time
	nsec = init_end.tv_nsec-init_start.tv_nsec;
	if(nsec < 0) {
		nsec = 1E9+nsec;
		init_end.tv_sec-=1;
	}
	printf("Init time: %.9f\n", (double)(init_end.tv_sec-init_start.tv_sec)+(double)(nsec*1E-9));
	//Leaving variable computation time
	nsec = lv_end.tv_nsec-lv_start.tv_nsec;
	if(nsec < 0) {
		nsec = 1E9+nsec;
		lv_end.tv_sec-=1;
	}
	printf("Leaving variable computation time: %.9f\n", (double)(lv_end.tv_sec-lv_start.tv_sec)+(double)(nsec*1E-9));
	//Binv updating time
	nsec = b_end.tv_nsec-b_start.tv_nsec;
	if(nsec < 0) {
		nsec = 1E9+nsec;
		b_end.tv_sec-=1;
	}
	printf("Binv updating time: %.9f\n", (double)(b_end.tv_sec-b_start.tv_sec)+(double)(nsec*1E-9));
	return 0;
}

void help() {
	printf("Input format:\n");
	printf("M N <vector c> <vector b> <matrix A>\n");
}
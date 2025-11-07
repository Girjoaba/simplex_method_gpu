#include <_time.h>
#include <stdio.h>
#include "culiblp.h"
#include "cumatman.h"

int kn, km, km1;
int *devidx;
float *devred, *devtemp;

// test stuff
float *tmm, *tmn, *tm1n, *t1m, *t1n, *t1m1;
int *it1m;
// test stuff
float lpsolve(float *A, float *b, float *c, float *xb, int *bi, int m, int n) {
	int i, opt;
	cublasStatus stat;
	float *devc, *devA, *devb;
	// Binv: Basis matrix inverse
	// newBinv: temporary matrix inverse for swap purposes
	// E: used to compute the inversion using just one mm multiplication
	// newBinv = E * Binv
	float *devBinv, *devnewBinv, *devE;
	// e: cost contributions vector used to determine the entering variable
	// D, y, yb: arrays used to compute the cost contributions vector
	// D = [-c ; A] y = cb * Binv yb = [1 y] e = yb * D
	float *devD, *devy, *devyb, *deve;
	// xb: current basis
	// cb: basis costs
	// xb = Binv * b
	float *devcb, *devxb;
	// A_e: entering variable column of constraint factors
	// alpha: the pivotal vector used to determine the leaving variable
	// theta: Increases vector
	// alpha = Binv * A_e
	float *devA_e, *devalpha, *devtheta;
	// Vector of flags indicating valid increases
	// (valid alpha[i]) <==> (theta_flag[i] == 1)
	int *devtheta_flag;
	// Vector containing basis variables’ indexes
	int *devbi;
	//Counter for unbounded solution checking
	int *devnum_max;
	// Indexes of the entering and leaving variables.
	int ei, li;
	// Cost to optimize
	// z = c * x
	float z;

	//Proper dimensions for kernel grids
	kn = (int)ceil((float)n/BS);
	km = (int)ceil((float)m/BS);
	km1 = (int)ceil((float)(m+1)/BS);

	//CUBLAS initialization
	clock_gettime(CLOCK_REALTIME, &alloc_start);
	stat = cublasInit();
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("Device memory allocation failed.\n");
		return 1;
	}
	// c
	stat = cublasAlloc(n, sizeof(*c), (void **)&devc);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else
			printf("Error in allocation.\n");
		return 1;
	}
	// b
	stat = cublasAlloc(m, sizeof(*b), (void **)&devb);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else
			printf("Error in allocation.\n");
		return 1;
	}
	// A
	stat = cublasAlloc(m*n, sizeof(*A), (void **)&devA);
	if(stat != CUBLAS_STATUS_SUCCESS) {
	if(stat == CUBLAS_STATUS_ALLOC_FAILED)
		printf("Memory allocation failed: lack of resources.\n");
	else printf("Error in allocation.\n");
		return 1;
	}
	// Binv, newBinv, E
	stat = cublasAlloc(m*m, sizeof(*devBinv), (void **)&devBinv);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else
		printf("Error in allocation.\n");
		return 1;
	}
	stat = cublasAlloc(m*m, sizeof(*devnewBinv), (void **)&devnewBinv);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else
			printf("Error in allocation.\n");
		return 1;
	}
	stat = cublasAlloc(m*m, sizeof(*devE), (void **)&devE);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else printf("Error in allocation.\n");
			return 1;
	}
	// D, y, yb, e
	stat = cublasAlloc((m+1)*n, sizeof(*devD), (void **)&devD);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else
			printf("Error in allocation.\n");
		return 1;
	}
	stat = cublasAlloc(m, sizeof(*devy), (void **)&devy);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED
			printf("Memory allocation failed: lack of resources.\n");
		else printf("Error in allocation.\n");
			return 1;
	}
	stat = cublasAlloc(m+1, sizeof(*devyb), (void **)&devyb);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else
		printf("Error in allocation.\n");
		return 1;
	}
	stat = cublasAlloc(n, sizeof(*deve), (void **)&deve);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else
			printf("Error in allocation.\n");
		return 1;
	}
	// cb, xb
	stat = cublasAlloc(m, sizeof(*devcb), (void **)&devcb);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else
			printf("Error in allocation.\n");
		return 1;
	}
	stat = cublasAlloc(n, sizeof(*devxb), (void **)&devxb);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else
			printf("Error in allocation.\n");
		return 1;
	}
	// A_e, alpha, theta
	stat = cublasAlloc(m, sizeof(*devA_e), (void **)&devA_e);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else
			printf("Error in allocation.\n");
		return 1;
	}
	stat = cublasAlloc(m, sizeof(*devalpha), (void **)&devalpha);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else
			printf("Error in allocation.\n");
		return 1;
	}
	stat = cublasAlloc(m, sizeof(*devtheta), (void **)&devtheta);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else
			printf("Error in allocation.\n");
		return 1;
	}
	// red, temp, idx
	stat = cublasAlloc(km, sizeof(*devred), (void **)&devred);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else
			printf("Error in allocation.\n");
		return 1;
	}
	stat = cublasAlloc(km, sizeof(*devtemp), (void **)&devtemp);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else
			printf("Error in allocation.\n");
		return 1;
	}
	stat = cublasAlloc(1, sizeof(*devidx), (void **)&devidx);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else
			printf("Error in allocation.\n");
		return 1;
	}
	// num_max
	stat = cublasAlloc(1, sizeof(*devnum_max), (void **)&devnum_max);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else
			printf("Error in allocation.\n");
		return 1;
	}
	// theta_flag & bi
	stat = cublasAlloc(m, sizeof(*devtheta_flag), (void **)&devtheta_flag);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else
			printf("Error in allocation.\n");
		return 1;
	}
	stat = cublasAlloc(m, sizeof(*devbi), (void **)&devbi);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_ALLOC_FAILED)
			printf("Memory allocation failed: lack of resources.\n");
		else
			printf("Error in allocation.\n");
		return 1;
	}
	clock_gettime(CLOCK_REALTIME, &alloc_end);
	
	clock_gettime(CLOCK_REALTIME, &init_start);
	//Move A,b,c(,yb,D) on device
	stat = cublasSetMatrix(m, n, sizeof(*A), A, m, devA, m);
	if(stat != CUBLAS_STATUS_SUCCESS){
		if(stat == CUBLAS_STATUS_MAPPING_ERROR)
			printf("Error accessing device memory.\n");
		else
			printf("Setting error.\n");
		return 1;
	}
	stat = cublasSetVector(m, sizeof(*b), b, 1, devb, 1);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_MAPPING_ERROR)
			printf("Error accessing device memory.\n");
		else
			printf("Setting error.\n");
		return 1;
	}
	stat = cublasSetVector(n, sizeof(*c), c, 1, devc, 1);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		if(stat == CUBLAS_STATUS_MAPPING_ERROR)
			printf("Error accessing device memory.\n");
		else
			printf("Setting error.\n");
		return 1;
	}

	//Initialize yb
	zeros<<<km1, BS>>>(devyb, 1, m);
	init_yb<<<1, BS>>>(devyb);
	//Initialize D
	init_cInD<<<kn, BS>>>(devc, devD, m+1, n);
	init_AInD<<<dim3(kn, km1), dim3(BS, BS)>>>(devA, devD, m, n);
	//Initialize devBinv <- Im
	init_I<<<dim3(km, km), dim3(BS, BS)>>>(devBinv, m);
	//devcb <- devc[n-m] to devc[n]
	cublasScopy(m, &devc[n-m], 1, devcb, 1);
	//devxb <- devb
	cublasScopy(m, devb, 1, devxb, 1);
	//devbi[i] = (n-m)+i
	init_bi<<<km, BS>>>(devbi, m, n);
	clock_gettime(CLOCK_REALTIME, &init_end);

	i=0;
	do {
		clock_gettime(CLOCK_REALTIME, &ev_start);
		// y = cb*Binv
		cublasSgemm('N', 'N', 1, m, m, 1.0f, devcb, 1, devBinv, m, 0.0f, devy, 1);
		cublasScopy(m, devy, 1, &devyb[1], 1);
		// e = [1 y]*[-c ; A]
		cublasSgemm('N', 'N', 1, n, m+1, 1.0f, devyb, 1, devD, m+1, 0.0f, deve, 1);
		clock_gettime(CLOCK_REALTIME, &blas_end);

		ei = entering_index(deve, n);
		clock_gettime(CLOCK_REALTIME, &ev_end);
		if (ei < 0) {
			opt = 1;
			break;
		}

		// alpha = Binv*A_e
		clock_gettime(CLOCK_REALTIME, &lv_start);
		extract_column(devA, devA_e, ei, n, m);
		cublasSgemv('N', m, m, 1.0f, devBinv, m, devA_e, 1, 0.0f, devalpha, 1);
		int num_max;
		cudaMemset(devnum_max, 0, 1);
		compute_theta<<<km, BS>>>(devxb, devalpha, devtheta, devtheta_flag, m, devnum_max);
		cudaMemcpy(&num_max, devnum_max, sizeof(int), cudaMemcpyDeviceToHost);
		if(num_max == m) {
			opt = 2;
			break;
		}
		li = leaving_index(devtheta, devtheta_flag, m);
		clock_gettime(CLOCK_REALTIME, &lv_end);

		clock_gettime(CLOCK_REALTIME, &b_start);
		if(compute_E(devE, devalpha, m, li)) {
			opt = 3;
			break;
		}
		// Binv = E*Binv
		cublasSgemm('N', 'N', m, m, m, 1.0f, devE, m, devBinv, m, 0.0f, devnewBinv, m);
		cublasScopy(m*m, devnewBinv, 1, devBinv, 1);
		clock_gettime(CLOCK_REALTIME, &b_end);
		
		//bi[lv] = ev;
		//cb[lv] = c[ev];
		update_bi_cb<<<km, BS>>>(devbi, devcb, devc, li, ei);
		// xb=Binv*b
		cublasSgemv('N', m, m, 1.0f, devBinv, m, devb, 1, 0.0f, devxb, 1);
		i++;
	} while(i<MAX_ITER);

	if(opt == 1) {
		z = cublasSdot(m, devcb, 1, devxb, 1);
		stat = cublasGetVector(m, sizeof(*devxb), devxb, 1, xb, 1);
		if(stat != CUBLAS_STATUS_SUCCESS) {
			if(stat == CUBLAS_STATUS_MAPPING_ERROR)
				printf("Error accessing device memory.\n");
			else
				printf("Setting error.\n");
			return 1;
		}
		stat = cublasGetVector(m, sizeof(*devbi), devbi, 1, bi, 1);
		if(stat != CUBLAS_STATUS_SUCCESS) {
			if(stat == CUBLAS_STATUS_MAPPING_ERROR)
				printf("Error accessing device memory.\n");
			else
				printf("Setting error.\n");
			return 1;
		}
	} else if(opt == 2)
		z = INFINITY;
	else
		z = NAN;
	
	clock_gettime(CLOCK_REALTIME, &dealloc_start);
	cublasFree(devc);
	cublasFree(devb);
	cublasFree(devA);
	cublasFree(devBinv);
	cublasFree(devnewBinv);
	cublasFree(devE);
	cublasFree(devD);
	cublasFree(devy);
	cublasFree(devyb);
	cublasFree(deve);
	cublasFree(devcb);
	cublasFree(devxb);
	cublasFree(devA_e);
	cublasFree(devalpha);
	cublasFree(devtheta);
	cublasFree(devnum_max);
	cublasFree(devtheta_flag);
	cublasFree(devbi);
	cublasFree(devidx);
	cublasFree(devtemp);
	cublasFree(devred);
	cublasShutdown();
	clock_gettime(CLOCK_REALTIME, &dealloc_end);
	return z;
}

/************** WRAPPERS ***********************/
int entering_index(float *e, int n) {
	float val_min;
	int min_i = get_min_idx(e, n, &val_min);
	return (val_min >= -EPS) ? -1 : min_i;
}

void extract_column(float *M, float *v, int start_i, int stride, int size) {
	cublasScopy(size, &M[R2C(0,start_i,size)], 1, v, 1);
}

int leaving_index(float *t, int *flag, int size) {
	return get_min_idx(t, size, NULL);
}

int compute_E(float *E, float *alpha, int m, int li) {
	float qth, *devqth; // = a[li];
	cudaMalloc((void **)&devqth, sizeof(float));
	get_val<<<km, BS>>>(alpha, li, devqth);
	cudaMemcpy(&qth, devqth, sizeof(float), cudaMemcpyDeviceToHost);
	if((qth >= -EPS) && (qth <= EPS)) {
		printf("qth == 0....exit...\n");
		return 1;
	}
	init_I<<<dim3(km, km), dim3(BS, BS)>>>(E, m);
	compute_new_E<<<km, BS>>>(E, alpha, m, li, qth);
	return 0;
}

int get_min_idx(float *a, int n, float *val) {
	int numBlocks = (int)ceil((float)n/BS);
	int size = n;
	int min_idx = -1;
	cublasScopy(size, a, 1, devtemp, 1);
	do {
		reduce_min<<<numBlocks, BS>>>(devtemp, size, devred);
		size = numBlocks;
		if(numBlocks > 1) {
			cublasScopy(size, devred, 1, devtemp, 1);
			numBlocks = (int)ceil((float)numBlocks/BS);
		}
	} while(size > 1);

	numBlocks = (int)ceil((float)n/BS);
	get_idx<<<numBlocks, BS>>>(a, devidx, devred, n);
	cudaMemcpy(&min_idx, devidx, sizeof(int), cudaMemcpyDeviceToHost);
	if(val != NULL)
		cudaMemcpy(val, devred, sizeof(float), cudaMemcpyDeviceToHost);
		 
	return min_idx;
}

/************* KERNELS ***********************/
__global__ void zeros(float *a, int m, int n) {
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int s = gridDim.x*blockDim.x;
	int id = AT(i,j,s);
	if(id<m*n) {
		i = id/n;
		j = id%n;
		a[R2C(i,j,m)] = 0;
	}
}

__global__ void reduce_min(float *f, int n, float *min) {
	int tid = threadIdx.x;
	int j = blockIdx.x*blockDim.x + tid;
	//Each block loads its elements into shared mem,
	//padding if not multiple of BS
	__shared__ float sf[BS];
	sf[tid] = (j<n) ? f[j] : FLT_MAX;
	__syncthreads();

	//Apply reduction
	for(int s=blockDim.x/2; s>0; s>>=1) {
		if(tid < s)
			sf[tid] = sf[tid] > sf[tid+s] ? sf[tid+s] : sf[tid];
			__syncthreads();
	}
	if(tid == 0)
		min[blockIdx.x] = sf[0];
}

__global__ void get_val(float *f, int index, float *val) {
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	if(j == index) *val = f[j];
}

__global__ void get_idx(float *f, int *index, float *val, int n) {
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	if(j == 0)
		index[0] = -1;
	__syncthreads();
	if(j < n) {
		float diff = f[j]-val[0];
		if(diff>=-EPS && diff<=EPS) atomicCAS(index, -1, j);
	}
}

__global__ void init_yb(float *yb) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i == 0) yb[0] = 1;
}

__global__ void init_cInD(float *c, float *D, int m, int n) {
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int s = gridDim.x*blockDim.x;
	int id = AT(i,j,s);

	if(id<n) {
		i = id/n;
		j = id%n;
		D[R2C(i,j,m)] = -c[id];
	}
}

__global__ void init_AInD(float *A, float *D, int m, int n) {
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int s = gridDim.x*blockDim.x;
	int id = AT(i,j,s);
	if(id<m*n) {
		i = id/n;
		j = id%n;
		D[R2C(i+1,j,m+1)] = A[R2C(i,j,m)];
	}
}
__global__ void init_I(float *I, int m) {
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int s = gridDim.x*blockDim.x;
	int id = AT(i,j,s);
	if(id<m*m) {
		i = id/m;
		j = id%m;
		I[R2C(i,j,m)] = (float)(i==j);
	}
}

__global__ void init_bi(int *bi, int m, int n) {
int i = blockIdx.y*blockDim.y + threadIdx.y;
int j = blockIdx.x*blockDim.x + threadIdx.x;
int s = gridDim.x*blockDim.x;

int id = AT(i,j,s);
if(id<m)
	bi[id] = (n-m)+id;
}

//num_max counts how many alpha[i] are <= 0
__global__ void compute_theta(float *xb, float *alpha, float *theta, int *theta_flag, int m, int *num_max) {
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int s = gridDim.x*blockDim.x;

	int id = AT(i,j,s);

	if (id<m) {
		int cond = (alpha[id]>0);
		theta_flag[id]= cond;
		theta[id]=xb[id]/alpha[id]*cond + FLT_MAX*(1-cond);
		atomicAdd(num_max, 1-cond);
	}
}

__global__ void compute_new_E(float *E, float *alpha, int m, int li, float qth) {
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int s = gridDim.x*blockDim.x;
	int id = AT(i,j,s);
	if(id<m) {
		alpha[id] = -alpha[id]/qth;
		if(id==li)
			alpha[id]=1/qth;
		E[R2C(id, li, m)] = alpha[id];
	}
}

__global__ void update_bi_cb(int *bi, float *cb, float *c, int li, int ei) {
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	//bi[lv] = ev;
	//cb[lv] = c[ev];
	if (j == li) {
		bi[j] = ei;
		cb[j] = c[ei];
	}
}

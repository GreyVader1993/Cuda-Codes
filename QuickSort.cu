/*
 ============================================================================
 Name        : cuda.c
 Author      :
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

 #include<stdio.h>
#include <cuda.h>
#include<cuda_runtime.h>

 __global__ static void quicksort(int* values,int N) {
 #define MAX_LEVELS	300

	int pivot, L, R;
	int idx =  threadIdx.x + blockIdx.x * blockDim.x;
	int start[MAX_LEVELS];
	int end[MAX_LEVELS];

	start[idx] = idx;
	end[idx] = N - 1;
	while (idx >= 0) {
		L = start[idx];
		R = end[idx];
		if (L < R) {
			pivot = values[L];
			while (L < R) {
				while (values[R] >= pivot && L < R)
					R--;
				if(L < R)
					values[L++] = values[R];
				while (values[L] < pivot && L < R)
					L++;
				if (L < R)
					values[R--] = values[L];
			}
			values[L] = pivot;
			start[idx + 1] = L + 1;
			end[idx + 1] = end[idx];
			end[idx++] = L;
			if (end[idx] - start[idx] > end[idx - 1] - start[idx - 1]) {
	                        // swap start[idx] and start[idx-1]
        	                int tmp = start[idx];
                	        start[idx] = start[idx - 1];
                        	start[idx - 1] = tmp;

	                        // swap end[idx] and end[idx-1]
        	                tmp = end[idx];
                	        end[idx] = end[idx - 1];
                        	end[idx - 1] = tmp;
	                }

		}
		else
			idx--;
	}
}


int main(){
  int x[20],size,i;
  int *d_x,*d_size,*d_i;

  printf("Enter size of the array: ");
  scanf("%d",&size);

  printf("Enter %d elements: ",size);
  for(i=0;i<size;i++)
    scanf("%d",&x[i]);
  	cudaMalloc((void **)&d_x,sizeof(int)*size);
    cudaMalloc((void **)&d_size,sizeof(int));
    cudaMalloc((void **)&d_i,sizeof(int));

  cudaMemcpy(d_x, &x,  sizeof( int)*size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_size, &size,  sizeof( int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_i, &i,  sizeof( int), cudaMemcpyHostToDevice);

  quicksort<<<1,1,size>>>(d_x,size);
  cudaMemcpy(x, d_x, sizeof(int)*size, cudaMemcpyDeviceToHost);
  printf("Sorted elements: ");
  for(i=0;i<size;i++)
    printf(" %d",x[i]);

 cudaFree(d_x);
 cudaFree(d_size);
 cudaFree(d_i);



  return 0;
}


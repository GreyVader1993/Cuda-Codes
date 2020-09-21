#include<stdio.h>
#include<cuda.h>
/*__global__ void device_func(int *a)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

		a[tid]+=1;


	//int tmp;
for(int i=0;i<6;i++)
{
if(i%2==0)
{

even(a)
/*tmp=a[threadIdx.x*2];
a[threadIdx.x]=a[threadIdx.x*2+1];
a[threadIdx.x*2+1]=tmp;
	}
else{
	odd(a)

	/*tmp=a[threadIdx.x*2+1];
	a[threadIdx.x+1]=a[threadIdx.x*2+2];
	a[threadIdx.x*2+2]=tmp;

}}*/
__global__ void even(int *a)
{
	if(a[threadIdx.x*2]>a[threadIdx.x*2+1]){
	int
	tmp=a[threadIdx.x*2];
	a[threadIdx.x*2]=a[threadIdx.x*2+1];
	a[threadIdx.x*2+1]=tmp;}
	__syncthreads();
}
__global__ void odd(int *a)
{
	if(a[threadIdx.x*2+1]>a[threadIdx.x*2+2])
{int tmp;
	tmp=a[threadIdx.x*2+1];
		a[threadIdx.x*2+1]=a[threadIdx.x*2+2];
		a[threadIdx.x*2+2]=tmp;
		__syncthreads();
}}




int main()
{
int a[]={6,7,5,2,3,1};
int *devA;
int c[6];
printf("hello");
//c=(int *)malloc(sizeof(int)*6);
cudaMalloc((void **)&devA,sizeof(int)*6);
cudaMemcpy(devA,a,sizeof(int)*6,cudaMemcpyHostToDevice);

for(int i=0;i<6;i++)
{
if(i%2==0)
{
	even<<<1,3>>>(devA);
}
else{

	odd<<<1,3>>>(devA);
}
}
//device_func<<<1,3>>>(devA);
cudaMemcpy(c,devA,sizeof(int)*6,cudaMemcpyDeviceToHost);
printf("After sort--------\n");
for(int j=0;j<6;j++)
{
printf("%d ",c[j]);
}
cudaFree(devA);
return 0;
}

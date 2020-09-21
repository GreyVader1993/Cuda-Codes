#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>
#include<cuda_runtime_api.h>
#define Tile_size 3
#define funcCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            printf( "Failed to run stmt %d ", __LINE__);                       \
            printf( "Got CUDA error ...  %s ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

int i,j,k;
int Row_A,Row_B,Row_C;
int Col_A,Col_B,Col_C;


//#########################################################
void MatrixMultonHost(float * A,float * B, float * C)
{

	for (int i=0; i < Row_A; i ++)
	    {
	        for (int j = 0; j < Col_A; j++)
	        {
	            C[i*Col_C + j ] = 0.0;
	            for (int k = 0; k < Col_C; k++)
	            {
	                C[i*Col_C + j ] += A[i*Col_A + k] * B [k*Col_C + j];
	            }
	        }//Second For close
	    }//First For close

}//Function close
//#########################################################
void Print_Mat(int Row,int Col,float * Mat)
{
	for(int i=0;i<Row*Col;i++)
			{
			printf("%f  ",*(Mat+i));

			if((i%Col)==0 )
				{
					printf("\n");
				}
			}
}//Function close
//#########################################################
__global__ void MatrixMultonDevice(float * A,float * B, float * C,int Row_A,int Col_A,int Col_C)
{
	int Row = blockDim.y*blockIdx.y + threadIdx.y;//Calculate id of thread.
	int Col = blockDim.x*blockIdx.x + threadIdx.x;
	float CValue;
	if((Row<Row_A)&&(Col<Col_C))
	{
		//float CValue=0.0;
		for(int i=0;i<Col_A;i++)
		{
			CValue+=A[Row*Col_A+i]*B[Col+i*Col_C];
		}
		C[Row*Col_C+Col]=CValue;
	}

}
/*//Copied
__global__ void matrixMultiply(float * A, float * B, float * C,
                   int numARows, int numAColumns,
                   int numBRows, int numBColumns,
                   int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here

	int Row = blockDim.y*blockIdx.y + threadIdx.y;
	int Col = blockDim.x*blockIdx.x + threadIdx.x;
	float tempSum = 0.0;

	if (Row < numARows && Col < numBColumns)
	{
		for(int j = 0; j < numAColumns; j++)
		{
			tempSum += A[j + Row*numAColumns] * B[j*numBColumns + Col];
		}
		C[Row*numCColumns + Col] = tempSum;
	}
}
*/

int main()
{

	float * A;
	float * B;
	float * C;
	float * Dev_A;
	float * Dev_B;
	float * Dev_C;
	float * DeviceComputed_C;

	printf("\nPlease Enter Rows and Columns of A:");
	scanf("%d %d",&Row_A,&Col_A);

	printf("\nPlease Enter Rows and Columns of B:");
	scanf("%d %d",&Row_B,&Col_B);
	/*//Matrix A initialization
	for(int i=0;i<Row;i++)
	{
		for(int j=0;j<Col;j++)
		{
		A[i][j]=1.0;
		}
	}*/

	A = (float *) malloc(sizeof(float)*Row_A*Col_A);
	B = (float *) malloc(sizeof(float)*Row_B*Col_B);

	//Matrix Initialization
	for(int i=0;i<Row_A*Col_A;i++)
	{
	 A[i]=1.0;

	}
	for(int i=0;i<Row_B*Col_B;i++)
	{

		 B[i]=1.0;
	}
	/*for(int i=0;i<Row*Col;i++)
		{
		 B[i]=1.0;
		}*/
	//*(A+0)=1.0;*(A+4)=4.0;
	//Printing Matrix A
	/*for(int i=0;i<Row;i++)
		{
			for(int j=0;j<Col;j++)
			{
			//printf("%f  ",A[i][j]);
			}printf("\n");
		}*/

	//Printing Matrix B
printf("\nMatrix A Values:\n");
Print_Mat(Row_A,Col_A,A);//Function Call

printf("\n\nMatrix B Values:\n");
Print_Mat(Row_B,Col_B,B);//Function Call


	cudaMalloc((void **)&Dev_A,sizeof(float)*Row_A*Col_A);
	cudaMalloc((void **)&Dev_B,sizeof(float)*Row_B*Col_B);
	cudaMemcpy(Dev_A,&A,sizeof(float)*Row_A*Col_A,cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_B,&B,sizeof(float)*Row_B*Col_B,cudaMemcpyHostToDevice);



//Matrix Multiplication on Host
	if(Col_A==Row_B)
		{
			Row_C=Row_A;
			Col_C=Col_B;
			C = (float *) malloc(sizeof(float)*Row_C*Col_C);
			MatrixMultonHost(A,B,C);//Function Call

			DeviceComputed_C=(float *)malloc(sizeof(float)*Row_C*Col_C);//Allocate Memory
			cudaMalloc((void **)&Dev_C,sizeof(float)*Row_C*Col_C);//Allocate Memory
			//cudaMemcpy(Dev_C,&DeviceComputed_C,sizeof(float)*Row_C*Col_C,cudaMemcpyHostToDevice);//Copy Memory to Device from Host
			//cudaMemcpy(Dev_C,&C,sizeof(float)*Row_C*Col_C,cudaMemcpyHostToDevice);//Copy Memory to Device from Host
			dim3 dimBlock(Tile_size, Tile_size, 1);
		    dim3 dimGrid((Col_C/Tile_size) + 1, (Row_C/Tile_size) + 1, 1);
			MatrixMultonDevice<<<dimGrid,dimBlock>>>(Dev_A,Dev_B,Dev_C,Row_A,Col_A,Col_C); //Kernel Launch
//*/
			//MatrixMultonDevice<<<1,1>>>(Dev_A,Dev_B,Dev_C,Row_A,Col_A,Col_C); //Kernel Launch
			cudaThreadSynchronize();
			cudaError_t err1 = cudaPeekAtLastError();
			cudaDeviceSynchronize();
			printf( "CUDA error ... %s \n", cudaGetErrorString(err1));
			//Copy Back the Memory From Device To Host
			funcCheck(cudaMemcpy(DeviceComputed_C,Dev_C,sizeof(float)*Row_C*Col_C,cudaMemcpyDeviceToHost));


		}
		else
		{
			printf("\n Matrix Multiplication can not be performed\n");
		}

//Matrix Multiplication on Device or GPU

	/*cudaMalloc((void **)&Dev_A,sizeof(float)*Row_A*Col_A);
	cudaMalloc((void **)&Dev_B,sizeof(float)*Row_B*Col_B);
	cudaMemcpy(Dev_A,&A,sizeof(float)*Row_A*Col_A,cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_B,&B,sizeof(float)*Row_B*Col_B,cudaMemcpyHostToDevice);

	/*if(Col_A==Row_B)
		{
			Row_C=Row_A;
			Col_C=Col_B;
			DeviceComputed_C=(float *)malloc(sizeof(float)*Row_C*Col_C);//Allocate Memory
			cudaMalloc((void **)&Dev_C,sizeof(float)*Row_C*Col_C);//Allocate Memory
			cudaMemcpy(Dev_C,&DeviceComputed_C,sizeof(float)*Row_C*Col_C,cudaMemcpyHostToDevice);//Copy Memory to Device from Host
			//cudaMemcpy(Dev_C,&C,sizeof(float)*Row_C*Col_C,cudaMemcpyHostToDevice);//Copy Memory to Device from Host
			/*dim3 dimBlock(16, 16, 1);
			dim3 dimGrid((Col_C / 16) + 1, (Row_C / 16) + 1, 1);

			MatrixMultonDevice<<<dimGrid,dimBlock>>>(Dev_A,Dev_B,Dev_C,Row_A,Col_A,Col_C); //Kernel Launch
			//**\/
			MatrixMultonDevice<<<1,1>>>(Dev_A,Dev_B,Dev_C,Row_A,Col_A,Col_C); //Kernel Launch
			cudaThreadSynchronize();
			cudaError_t err1 = cudaPeekAtLastError();
			cudaDeviceSynchronize();
			printf( "CUDA error ... %s \n", cudaGetErrorString(err1));


		}
		else
		{
			printf("\n Matrix Multiplication can not be performed\n");
		}*\/
//Copy Back the Memory From Device To Host
	funcCheck(cudaMemcpy(DeviceComputed_C,Dev_C,sizeof(float)*Row_C*Col_C,cudaMemcpyDeviceToHost));
*/


printf("\n\nResult Matrix C from Host:\n");
Print_Mat(Row_C,Col_C,C);//Function Call

printf("\n\nResult Matrix C from Device:\n");
Print_Mat(Row_C,Col_C,DeviceComputed_C);//Function Call

cudaFree(Dev_A);
cudaFree(Dev_B);
cudaFree(Dev_C);
free(A);
free(B);
free(C);
free(DeviceComputed_C);
return 0;
}

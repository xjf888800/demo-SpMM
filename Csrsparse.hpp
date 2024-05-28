#include <hip/hip_runtime.h>

__global__ void comput_U(int *d_U,
                         const csrIdxType*  dptr_offset_A,
                         const csrIdxType*  dptr_colindex_A,
                         const csrIdxType*  dptr_offset_B,
                         int m)
{     
        int x,j;  
        int tid=blockIdx.x * blockDim.x + threadIdx.x;
        if(tid<m){
                d_U[tid]=0;
                for(x=dptr_offset_A[tid];x<dptr_offset_A[tid+1];x++){
                        j=dptr_colindex_A[x];
                        d_U[tid]=d_U[tid]+(dptr_offset_B[j+1]-dptr_offset_B[j]);
                }
        }
}

__global__ void comput_tem_value(const csrIdxType*  dptr_offset_A,
                                 const csrIdxType*  dptr_colindex_A,
                                 const dtype*       dptr_value_A,
                                 const csrIdxType*  dptr_offset_B,
                                 const csrIdxType*  dptr_colindex_B,
                                 const dtype*       dptr_value_B,
                                 csrIdxType*        tem_pdptr_colindex_C,
                                 dtype*             tem_pdptr_value_C,
                                 csrIdxType*        dptr_offset_C,
                                 const dtype        alpha,
                                 int m)
{
        int x,j,l,r,index;  
        int tid=blockIdx.x * blockDim.x + threadIdx.x;
        if(tid<m){
                index=0;
                for(x=dptr_offset_A[tid];x<dptr_offset_A[tid+1];x++){
                        j=dptr_colindex_A[x];
                        for(l=dptr_offset_B[j]; l < dptr_offset_B[j+1]; l++){
                                r = dptr_colindex_B[l];
                                tem_pdptr_colindex_C[dptr_offset_C[tid]+index] = r;
                                tem_pdptr_value_C[dptr_offset_C[tid]+index]=alpha*dptr_value_A[x] * dptr_value_B[l]; //alpha*
                                index++;
                        }
                }
        }
}

__device__ void quick_sort(int *num, double *num1,int low, int high )
{
    int i,j,tmp,temp;
    double val,t;
    i = low;
    j = high;
    tmp = num[low];   //任命为中间分界线，左边比他小，右边比他大,通常第一个元素是基准数
	val=num1[low];
    if(i > j)  //如果下标i大于下标j，函数结束运行
    {
        return;
    }
    while(i != j)
    {
        while(num[j] >= tmp && j > i)   
        {
            j--;
        }
        while(num[i] <= tmp && j > i)
        {
            i++;
        }
        if(j > i)
        {
            temp = num[j];
            num[j] = num[i];
            num[i] = temp;
			t=num1[j];
			num1[j]=num1[i];
			num1[i]=t;
        }
    }
    num[low] = num[i];
    num[i] = tmp;
	num1[low]=num1[i];
	num1[i]=val;
    quick_sort(num,num1,low,i-1);
    quick_sort(num,num1,i+1,high);
}

__global__ void sort(csrIdxType*       tem_pdptr_colindex_C,
                dtype*            tem_pdptr_value_C,
                csrIdxType*        dptr_offset_C,
                int m)
{
        int tid=blockIdx.x * blockDim.x + threadIdx.x;
        if(tid<m){
                quick_sort(tem_pdptr_colindex_C,tem_pdptr_value_C,dptr_offset_C[tid],dptr_offset_C[tid+1]-1);
        }
}

__global__ void add_sameclo_value(csrIdxType*        dptr_offset_C,
                                  csrIdxType*        tem_pdptr_colindex_C,
                                  dtype*             tem_pdptr_value_C,
                                  int* d_U,
                                  int m,
                                  csrIdxType*        tem_colindex_C
                                  )
{
    int tid=blockIdx.x * blockDim.x + threadIdx.x;
    int x,temp,not_samecol;
    if(tid<m){
    	temp=dptr_offset_C[tid];
        if(dptr_offset_C[tid]==dptr_offset_C[tid+1]){
            not_samecol=0;
            d_U[tid]=not_samecol;
        }else{
            not_samecol=1;
        }
        for(x=dptr_offset_C[tid];x<dptr_offset_C[tid+1]-1;x++){
            if(tem_pdptr_colindex_C[x]==tem_pdptr_colindex_C[x+1]){
                tem_colindex_C[x+1]=-1;
                tem_pdptr_value_C[temp]=tem_pdptr_value_C[temp]+tem_pdptr_value_C[x+1];
            }
            else{
                temp=x+1;
                not_samecol++;
            }
        }
        d_U[tid]=not_samecol;
    }
}

__global__ void copy_data(csrIdxType* old_device_offsetC,
                    csrIdxType*        dptr_offset_C,
                    csrIdxType*        tem_colindex_C,
                    csrIdxType*        tem_pdptr_colindex_C,
                    dtype*             tem_pdptr_value_C,
                    csrIdxType**       pdptr_colindex_C,
		            dtype**            pdptr_value_C,
                    int m)
{
    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int index,x;
    if(i<m){
        index=0;
        for(x=old_device_offsetC[i];x<old_device_offsetC[i+1];x++){
            if(tem_colindex_C[x]!=-1){
                (*pdptr_colindex_C)[dptr_offset_C[i]+index]=tem_pdptr_colindex_C[x];
                (*pdptr_value_C)[dptr_offset_C[i]+index]=tem_pdptr_value_C[x];
                index=index+1;
            }
        }
    }
}

void  call_device_spgemm(const int transA,
		const int          transB,
		const dtype        alpha,
		const size_t       m,
		const size_t       n,
		const size_t       k,
                const size_t       nnz_A,
		const csrIdxType*  dptr_offset_A,
		const csrIdxType*  dptr_colindex_A,
		const dtype*       dptr_value_A,
                const size_t       nnz_B,
		const csrIdxType*  dptr_offset_B,
		const csrIdxType*  dptr_colindex_B,
		const dtype*       dptr_value_B,
        size_t*            ptr_nnz_C,
        csrIdxType*        dptr_offset_C,
        csrIdxType**       pdptr_colindex_C,
		dtype**            pdptr_value_C )
{
    // ...
    //计算d_U
    my_timer f_one,f_two,quik;
    f_one.start();
    int *d_U;
    HIP_CHECK( hipMalloc((void**)&d_U, m * sizeof(int)) )   //注意&
    csrIdxType*     old_device_offsetC; 
    HIP_CHECK( hipMalloc((void**) &old_device_offsetC, (m+1) * sizeof(csrIdxType)) )
    int threads_per_block=256;
    int num_blocks=(m+threads_per_block-1)/threads_per_block;
    comput_U<<<num_blocks,threads_per_block>>>(d_U,dptr_offset_A,dptr_colindex_A,dptr_offset_B,m);
    HIP_CHECK(hipDeviceSynchronize()); 	  //注意同步
    //for(int i=0;i<m;i++){
    	//printf("%d   \n",d_U[i]);
    //}//测试成功
    //
    size_t tem_nonzero = 0;
    dptr_offset_C[0]=0;
    old_device_offsetC[0]=0;
    for(int i=0;i<m;i++){
        tem_nonzero=tem_nonzero+d_U[i];
        dptr_offset_C[i+1]=tem_nonzero;
        old_device_offsetC[i+1]=tem_nonzero;
        //printf("%d   \n",dptr_offset_C[i+1]);
    }
    //分配临时空间
    size_t     tem_ptr_nnz_C=0;
    csrIdxType*        tem_pdptr_colindex_C;
    dtype*             tem_pdptr_value_C;
    tem_ptr_nnz_C=tem_nonzero;
    HIP_CHECK( hipMalloc((void**)&tem_pdptr_colindex_C, tem_ptr_nnz_C * sizeof(csrIdxType)) )
    HIP_CHECK( hipMalloc((void**)&tem_pdptr_value_C, tem_ptr_nnz_C * sizeof(dtype)) )
    //初步计算（不排序，不合并）
    comput_tem_value<<<num_blocks,threads_per_block>>>(dptr_offset_A,dptr_colindex_A,dptr_value_A,dptr_offset_B,dptr_colindex_B,dptr_value_B,
                                                        tem_pdptr_colindex_C,tem_pdptr_value_C,dptr_offset_C,alpha,m);
    HIP_CHECK(hipDeviceSynchronize()); //注意同步

    //按列索引排序
    csrIdxType*        tem_colindex_C;//合并的时候用
    HIP_CHECK( hipMalloc((void**)&tem_colindex_C, tem_ptr_nnz_C * sizeof(csrIdxType)) )
    sort<<<num_blocks,threads_per_block>>>(tem_pdptr_colindex_C,tem_pdptr_value_C,dptr_offset_C,m);
    HIP_CHECK(hipDeviceSynchronize()); //注意同步

    add_sameclo_value<<<num_blocks,threads_per_block>>>(dptr_offset_C,tem_pdptr_colindex_C,tem_pdptr_value_C,d_U,m,tem_colindex_C);
    HIP_CHECK(hipDeviceSynchronize()); //注意同步
    f_one.stop();
    cout << "f_one time: " << f_one.time_use << "(us)" << endl;
    

    quik.start();
     //更新dptr_offset_C
    int nnz=0;
    for(int i=0;i<m;i++){
        nnz=nnz+d_U[i];
        dptr_offset_C[i+1]=nnz;
        //printf("%d   \n",dptr_offset_C[i+1]);
    }
    
	*ptr_nnz_C = nnz;
    HIP_CHECK( hipMalloc((void**) pdptr_colindex_C, nnz * sizeof(csrIdxType)) )
    HIP_CHECK( hipMalloc((void**) pdptr_value_C, nnz * sizeof(dtype)) )
    int index=0;
    for(int i=0;i<tem_ptr_nnz_C;i++){
        if(tem_colindex_C[i]!=-1){
            (*pdptr_colindex_C)[index]=tem_pdptr_colindex_C[i];
            (*pdptr_value_C)[index]=tem_pdptr_value_C[i];
            index++;
        }
    }
    // for(int i=0;i<m;i++){
    //     int index=0;
    //     for(int x=old_device_offsetC[i];x<old_device_offsetC[i+1];x++){
    //         if(tem_colindex_C[x]!=-1){
    //             (*pdptr_colindex_C)[dptr_offset_C[i]+index]=tem_pdptr_colindex_C[x];
    //             (*pdptr_value_C)[dptr_offset_C[i]+index]=tem_pdptr_value_C[x];
    //             index++;
    //         }
    //     }
    // }
    // copy_data<<<num_blocks,threads_per_block>>>(old_device_offsetC,dptr_offset_C,tem_colindex_C,tem_pdptr_colindex_C,
    //                                             tem_pdptr_value_C,
    //                                             pdptr_colindex_C,pdptr_value_C,m);
    // HIP_CHECK(hipDeviceSynchronize()) //注意同步
    quik.stop();
    cout << "quic_time : " << quik.time_use << "(us)" << endl;
    HIP_CHECK( hipFree(tem_pdptr_colindex_C) )
    HIP_CHECK( hipFree(tem_pdptr_value_C) )
    HIP_CHECK( hipFree(tem_colindex_C) )
    HIP_CHECK( hipFree(old_device_offsetC) )
    HIP_CHECK( hipFree(d_U) )
}
 

  
  
   
   
     


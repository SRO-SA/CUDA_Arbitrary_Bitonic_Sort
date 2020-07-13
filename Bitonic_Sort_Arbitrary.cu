__global__
void bitonic_sort_step(int length, long * arr, int j, int k, bool dir)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i< length; i+=stride)
  {
    int ixj = i^j;
    if((ixj)>i) {
      if(dir == true)
      {      
        if((i&k)==0)
        {
          if(arr[i]>arr[ixj] && ixj<length)
          {
            long tmp = arr[i];
            arr[i] = arr[ixj];
            arr[ixj] = tmp;

          }
        }
        if((i&k)!=0)
        {
          if(arr[i]<=arr[ixj] && ixj<length)
          {
            long tmp = arr[i];
            arr[i] = arr[ixj];
            arr[ixj] = tmp;
          }        
        }
      }
      else
      {      
        if((i&k)!=0)
        {
          if(arr[i]>arr[ixj] && ixj<length)
          {
            long tmp = arr[i];
            arr[i] = arr[ixj];
            arr[ixj] = tmp;
            tmp = arr[length + i];
            arr[ROW1*length + i] = arr[ROW1*length + ixj];
            arr[ROW1*length + ixj] = tmp; 
          }
          else if(arr[i]==arr[ixj]){
            if(arr[ROW1*length + i]>arr[ROW1*length + ixj]){
              long tmp = arr[i];
              arr[i] = arr[ixj];
              arr[ixj] = tmp;
              tmp = arr[ROW1*length + i];
              arr[ROW1*length + i] = arr[ROW1*length + ixj];
              arr[ROW1*length + ixj] = tmp;  
            }
          }
        }
        if((i&k)==0)
        {
          if(arr[i]<=arr[ixj] && ixj<length)
          {
            long tmp = arr[i];
            arr[i] = arr[ixj];
            arr[ixj] = tmp;
            tmp = arr[ROW1*length + i];
            arr[ROW1*length + i] = arr[ROW1*length + ixj];
            arr[ROW1*length + ixj] = tmp;
          }
          else if(arr[i]==arr[ixj]){
            if(arr[ROW1*length + i]<arr[ROW1*length + ixj]){
              long tmp = arr[i];
              arr[i] = arr[ixj];
              arr[ixj] = tmp;
              tmp = arr[ROW1*length + i];
              arr[ROW1*length + i] = arr[ROW1*length + ixj];
              arr[ROW1*length + ixj] = tmp;      
              
            }
          }        
        }
      }
    }
  }
}

long * bitonic_sort(int length, int numBlock, long * arr)
{
  int nextP2 = length == 1 ? 1 : 1 << (32 - __lzcnt(length-1));
  long* cudaArr;
  cudaMallocManaged(&cudaArr, length*ROW2*sizeof(long));
  cudaMemcpy(cudaArr, arr, length*ROW2*sizeof(long), cudaMemcpyHostToDevice);
  for(int i=2; i<=nextP2; i=i<<1) {
    for(int j=i>>1; j>0; j=j>>1){
      int tmp = length - 1;
      int tmpxj = (tmp^j);
      bool accending = true;
      if(tmpxj>tmp){
        int dir = (i&tmp);
        if(dir != 0) accending = false;
      }
      bitonic_sort_step<<<numBlock, BLOCKSIZE>>>(length, cudaArr, j, i, accending);
      cudaDeviceSynchronize();
    }
  }
  return cudaArr;
}

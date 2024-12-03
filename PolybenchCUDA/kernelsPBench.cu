//2mm kernels
template<typename T>
__global__ void mm2_kernel1(int ni, int nj, int nk, int nl, T alpha, T beta, T *tmp, T *A, T *B)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i < ni) && (j < nj))
  { 
    tmp[i * nj + j] = 0;
    int k;
    for (k = 0; k < nk; k++)
    {
      tmp[i * nj + j] += alpha * A[i * nk + k] * B[k * nj + j];
    }
  }
}


template<typename T>
__global__ void mm2_kernel2(int ni, int nj, int nk, int nl, T alpha, T beta, T *tmp, T *C, T *D)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i < ni) && (j < nl))
  { 
    D[i * nl + j] *= beta;
    int k;
    for (k = 0; k < nj; k++)
    {
      D[i * nl + j] += tmp[i * nj + k] * C[k * nl + j];
    }
  }
}

//3mm kernels
template<typename T>
__global__ void mm3_kernel1(int ni, int nj, int nk, int nl, int nm, T *A, T *B, T *E)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i < ni) && (j < nj))
  {
    E[i * nj + j] = 0;
    int k;
    for(k=0; k < nk; k++)
    {
      E[i * nj + j] += A[i * nk + k] * B[k * nj + j];
    }
  }
}

  
template<typename T>
__global__ void mm3_kernel2(int ni, int nj, int nk, int nl, int nm, T *C, T *D, T *F)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i < nj) && (j < nl))
  {
    F[i * nl + j] = 0;
    int k;
    for(k=0; k < nm; k++)
    {
      F[i * nl + j] += C[i * nm + k] * D[k * nl +j];
    }
  }
}

  
template<typename T>
__global__ void mm3_kernel3(int ni, int nj, int nk, int nl, int nm, T *E, T *F, T *G)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i < ni) && (j < nl))
  {
    G[i * nl + j] = 0;
    int k;
    for(k=0; k < nj; k++)
    {
      G[i * nl + j] += E[i * nj + k] * F[k * nl + j];
    }
  }
}

//atax kernels
template<typename T>
__global__ void atax_kernel1(int nx, int ny, T *A, T *x, T *tmp)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nx)
  {
    tmp[i] = 0;
    int j;
    for(j=0; j < ny; j++)
    {
      tmp[i] += A[i*ny+j] * x[j];
    }
  }
}

template<typename T>
__global__ void atax_kernel2(int nx, int ny, T *A, T *y, T *tmp)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (j < ny)
  {
    y[j] = 0;
    int i;
    for(i=0; i < nx; i++)
    {
      y[j] += A[i*ny+j] * tmp[i];
    }
  }
}

//bicg kernels
//Distributed (split) from initial loop and permuted into reverse order to allow parallelism...
template<typename T>
__global__ void bicg_kernel1(int nx, int ny, T *A, T *r, T *s)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (j < ny)
  {
    s[j] = 0.0f;

    int i;
    for(i = 0; i < nx; i++)
    {
      s[j] += r[i] * A[i * ny + j];
    }
  }  
}


//Distributed (split) from initial loop to allow parallelism
template<typename T>
__global__ void bicg_kernel2(int nx, int ny, T *A, T *p, T *q)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < nx)
  {
    q[i] = 0.0f;

    int j;
    for(j=0; j < ny; j++)
    {
      q[i] += A[i * ny + j] * p[j];
    }
  }
}

//doitgen kernels
template<typename T>
__global__ void doitgen_kernel1(int nr, int nq, int np, T *sum, T *A, T *C4, int r)
{
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y * blockDim.y + threadIdx.y;

  if ((p < np) && (q < nq))
  {
    sum[r * (nq * np) + q * np + p] = (T)0.0;
  
    for (int s = 0; s < np; s++)
    {
      sum[r * (nq * np) + q * np + p] = sum[r * (nq * np) + q * np + p] + A[r * (nq * np) + q * np + s] * C4[s * np + p];
    }
  }
}

template<typename T>
__global__ void doitgen_kernel2(int nr, int nq, int np, T *sum, T *A, T *C4, int r)
{
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y * blockDim.y + threadIdx.y;

  if ((p < np) && (q < nq))
  {
    A[r * (nq * np) + q * np + p] = sum[r * (nq * np) + q * np + p];
  }
}

//gemm kernel
template<typename T>
__global__ void gemm_kernel(int ni, int nj, int nk, T alpha, T beta, T *a, T *b, T *c)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i < ni) && (j < nj))
  {  
    c[i * nj + j] *= beta;
    int k;
    for(k=0; k < nk; k++)
    {
      c[i * nj + j] += alpha * a[i * nk + k] * b[k * nj +j];
    }
  }
}

//gemver kernels
template<typename T>
__global__ void gemver_kernel1(int n, T alpha, T beta, T *a, T *v1, T *v2, T *u1, T *u2)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i < n) && (j < n))
  {
    a[i * n + j] += u1[i] * v1[j] + u2[i] * v2[j];
  }
}


template<typename T>
__global__ void gemver_kernel2(int n, T alpha, T beta, T *a, T *x, T *y, T *z)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    int j;
    for(j = 0; j < n; j++) 
    {
      x[i] += beta * a[j * n + i] * y[j];
    }
    x[i] += z[i];
  }
}


template<typename T>
__global__ void gemver_kernel3(int n, T alpha, T beta, T *a, T *x, T *w)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if ((i >= 0) && (i < n))
  {
    int j;
    for(j = 0; j < n; j++)
    { 
      w[i] += alpha * a[i*n + j] * x[j];
    }
  }
}

//gesummv kernel
template<typename T>
__global__ void gesummv_kernel(int n, T alpha, T beta, T* A, T* B, T* tmp, T* x, T* y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    int j;
    for(j = 0; j < n; j++)
    {  
      tmp[i] += A[i * n + j] * x[j];
      y[i] += B[i * n + j] * x[j];
    }
    y[i] = alpha * tmp[i] + beta  * y[i];
  }
}

//mvt kernels
template<typename T>
__global__ void mvt_kernel1(int n, T *a, T *x1, T *y_1)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    int j;
    for(j=0; j < n; j++)
    {
      x1[i] += a[i * n + j] * y_1[j];
    }
  }
}

template<typename T>
__global__ void mvt_kernel2(int n, T *a, T *x2, T *y_2)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    int j;
    for(j=0; j < n; j++)
    {
      x2[i] += a[j * n + i] * y_2[j];  
    }
  }
}

//syr2k kernel
template<typename T>
__global__ void syr2k_kernel(int ni, int nj, T alpha, T beta, T *a, T *b, T *c)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i < ni) && (j < ni))
  {
    c[i * ni + j] *= beta;
    
    int k;
    for(k = 0; k < nj; k++)
    {
      c[i * ni + j] += alpha * a[i * nj + k] * b[j * nj + k] + alpha * b[i * nj + k] * a[j * nj + k];
    }
  }
}

//syrk kernel
template<typename T>
__global__ void syrk_kernel(int ni, int nj, T alpha, T beta, T *a, T *c)
{
  /*  C := alpha*A*A' + beta*C */
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i < ni) && (j < ni))
  {
    c[i * ni + j] *= beta;
    int k;    
    for(k=0; k < nj; k++)
    {
      c[i * ni + j] += alpha * a[i * nj + k] * a[j * nj + k];
    }
  }
}
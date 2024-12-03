template<typename T>
void mm2Cuda(int ni, int nj, int nk, int nl, T alpha, T beta,
    T* tmp, T* A, T* B, T* C, T* D, T* D_outputFromGpu)
{
  T *tmp_gpu;
  T *A_gpu;
  T *B_gpu;
  T *C_gpu;
  T *D_gpu;

  cudaMalloc((void **)&tmp_gpu, sizeof(T) * ni * nj);
  cudaMalloc((void **)&A_gpu, sizeof(T) * ni * nk);
  cudaMalloc((void **)&B_gpu, sizeof(T) * nk * nj);
  cudaMalloc((void **)&C_gpu, sizeof(T) * nl * nj);
  cudaMalloc((void **)&D_gpu, sizeof(T) * ni * nl);
  
  cudaMemcpy(tmp_gpu, tmp, sizeof(T) * ni * nj, cudaMemcpyHostToDevice);
  cudaMemcpy(A_gpu, A, sizeof(T) * ni * nk, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, sizeof(T) * nk * nj, cudaMemcpyHostToDevice);
  cudaMemcpy(C_gpu, C, sizeof(T) * nl * nj, cudaMemcpyHostToDevice);
  cudaMemcpy(D_gpu, D, sizeof(T) * ni * nl, cudaMemcpyHostToDevice);  
    
  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid1((size_t)ceil( ((float)nj) / ((float)block.x) ), (size_t)ceil( ((float)ni) / ((float)block.y)) );
  dim3 grid2((size_t)ceil( ((float)nl) / ((float)block.x) ), (size_t)ceil( ((float)ni) / ((float)block.y)) );

  /* Start timer. */
    polybench_start_instruments;

  mm2_kernel1<<<grid1,block>>>(ni, nj, nk, nl, alpha, beta, tmp_gpu, A_gpu, B_gpu);
  cudaThreadSynchronize();
  mm2_kernel2<<<grid2,block>>>(ni, nj, nk, nl, alpha, beta, tmp_gpu, C_gpu, D_gpu);
  cudaThreadSynchronize();

  printf("GPU Time in seconds:\n");
    polybench_stop_instruments;
   polybench_print_instruments;

  cudaMemcpy(D_outputFromGpu, D_gpu, sizeof(T) * ni * nl, cudaMemcpyDeviceToHost);

  cudaFree(tmp_gpu);
  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
  cudaFree(D_gpu);
}

template<typename T>
void mm3Cuda(int ni, int nj, int nk, int nl, int nm,
    T* E,
    T* A,
    T* B,
    T* F,
    T* C,
    T* D,
    T* G,
    T* G_outputFromGpu)
{
  T *A_gpu;
  T *B_gpu;
  T *C_gpu;
  T *D_gpu;
  T *E_gpu;
  T *F_gpu;
  T *G_gpu;
  
  cudaMalloc((void **)&A_gpu, sizeof(T) * ni * nk);
  cudaMalloc((void **)&B_gpu, sizeof(T) * nk * nj);
  cudaMalloc((void **)&C_gpu, sizeof(T) * nj * nm);
  cudaMalloc((void **)&D_gpu, sizeof(T) * nm * nl);
  cudaMalloc((void **)&E_gpu, sizeof(T) * ni * nj);
  cudaMalloc((void **)&F_gpu, sizeof(T) * nj * nl);
  cudaMalloc((void **)&G_gpu, sizeof(T) * ni * nl);

  cudaMemcpy(A_gpu, A, sizeof(T) * ni * nk, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, sizeof(T) * nk * nj, cudaMemcpyHostToDevice);
  cudaMemcpy(C_gpu, C, sizeof(T) * nj * nm, cudaMemcpyHostToDevice);
  cudaMemcpy(D_gpu, D, sizeof(T) * nm * nl, cudaMemcpyHostToDevice);
  cudaMemcpy(E_gpu, E, sizeof(T) * ni * nj, cudaMemcpyHostToDevice);
  cudaMemcpy(F_gpu, F, sizeof(T) * nj * nl, cudaMemcpyHostToDevice);
  cudaMemcpy(G_gpu, G, sizeof(T) * ni * nl, cudaMemcpyHostToDevice);  
  
  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid1((size_t)(ceil( ((float)nj) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)ni/ ((float)DIM_THREAD_BLOCK_Y) )));
  dim3 grid2((size_t)(ceil( ((float)nl) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)nj/ ((float)DIM_THREAD_BLOCK_Y) )));
  dim3 grid3((size_t)(ceil( ((float)nl) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)ni/ ((float)DIM_THREAD_BLOCK_Y) )));

  /* Start timer. */
    polybench_start_instruments;

  mm3_kernel1<<<grid1,block>>>(ni, nj, nk, nl, nm, A_gpu, B_gpu, E_gpu);
  cudaThreadSynchronize();
  mm3_kernel2<<<grid2,block>>>(ni, nj, nk, nl, nm, C_gpu, D_gpu, F_gpu);
  cudaThreadSynchronize();
  mm3_kernel3<<<grid3,block>>>(ni, nj, nk, nl, nm, E_gpu, F_gpu, G_gpu);
  cudaThreadSynchronize();

  /* Stop and print timer. */
  printf("GPU Time in seconds:\n");
    polybench_stop_instruments;
   polybench_print_instruments;
  cudaMemcpy(G_outputFromGpu, G_gpu, sizeof(T) * ni * nl, cudaMemcpyDeviceToHost);
  
  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
  cudaFree(D_gpu);
  cudaFree(E_gpu);
  cudaFree(F_gpu);
  cudaFree(G_gpu);
}

template<typename T>
void ataxGpu(int nx, int ny, T* A, T* x, T* y, 
    T* tmp, T* y_outputFromGpu)
{
  T *A_gpu;
  T *x_gpu;
  T *y_gpu;
  T *tmp_gpu;

  cudaMalloc((void **)&A_gpu, sizeof(T) * nx * ny);
  cudaMalloc((void **)&x_gpu, sizeof(T) * ny);
  cudaMalloc((void **)&y_gpu, sizeof(T) * ny);
  cudaMalloc((void **)&tmp_gpu, sizeof(T) * nx);
  
  cudaMemcpy(A_gpu, A, sizeof(T) * nx * ny, cudaMemcpyHostToDevice);
  cudaMemcpy(x_gpu, x, sizeof(T) * ny, cudaMemcpyHostToDevice);
  cudaMemcpy(y_gpu, y, sizeof(T) * ny, cudaMemcpyHostToDevice);
  cudaMemcpy(tmp_gpu, tmp, sizeof(T) * nx, cudaMemcpyHostToDevice);
  
  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid1((size_t)(ceil( ((float)nx) / ((float)block.x) )), 1);
  dim3 grid2((size_t)(ceil( ((float)ny) / ((float)block.x) )), 1);

  /* Start timer. */
    polybench_start_instruments;

  atax_kernel1<<< grid1, block >>>(nx, ny, A_gpu,x_gpu,tmp_gpu);
  cudaThreadSynchronize();
  atax_kernel2<<< grid2, block >>>(nx, ny, A_gpu,y_gpu,tmp_gpu);
  cudaThreadSynchronize();
  
  /* Stop and print timer. */
  printf("GPU Time in seconds:\n");
    polybench_stop_instruments;
   polybench_print_instruments;
  
  cudaMemcpy(y_outputFromGpu, y_gpu, sizeof(T) * nx, cudaMemcpyDeviceToHost);

  cudaFree(A_gpu);
  cudaFree(x_gpu);
  cudaFree(y_gpu);
  cudaFree(tmp_gpu);
}

template<typename T>
void bicgCuda(int nx, int ny, T* A, T* r, T* s, 
  T* p, T* q, T* s_outputFromGpu, 
  T* q_outputFromGpu)
{
  T *A_gpu;
  T *q_gpu;
  T *p_gpu;
  T *r_gpu;
  T *s_gpu;

  cudaMalloc((void **)&A_gpu, sizeof(T) * nx * ny);
  cudaMalloc((void **)&r_gpu, sizeof(T) * nx);
  cudaMalloc((void **)&s_gpu, sizeof(T) * ny);
  cudaMalloc((void **)&p_gpu, sizeof(T) * ny);
  cudaMalloc((void **)&q_gpu, sizeof(T) * nx);
  cudaMemcpy(A_gpu, A, sizeof(T) * nx * ny, cudaMemcpyHostToDevice);
  cudaMemcpy(r_gpu, r, sizeof(T) * nx, cudaMemcpyHostToDevice);
  cudaMemcpy(s_gpu, s, sizeof(T) * ny, cudaMemcpyHostToDevice);
  cudaMemcpy(p_gpu, p, sizeof(T) * ny, cudaMemcpyHostToDevice);
  cudaMemcpy(q_gpu, q, sizeof(T) * nx, cudaMemcpyHostToDevice);

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid1((size_t)(ceil( ((float)ny) / ((float)block.x) )), 1);
  dim3 grid2((size_t)(ceil( ((float)nx) / ((float)block.x) )), 1);

  /* Start timer. */
    polybench_start_instruments;

  bicg_kernel1<<< grid1, block >>>(nx, ny, A_gpu, r_gpu, s_gpu);
  cudaThreadSynchronize();
  bicg_kernel2<<< grid2, block >>>(nx, ny, A_gpu, p_gpu, q_gpu);
  cudaThreadSynchronize();

  /* Stop and print timer. */
  printf("GPU Time in seconds:\n");
    polybench_stop_instruments;
   polybench_print_instruments;
  
  cudaMemcpy(s_outputFromGpu, s_gpu, sizeof(T) * ny, cudaMemcpyDeviceToHost);
  cudaMemcpy(q_outputFromGpu, q_gpu, sizeof(T) * nx, cudaMemcpyDeviceToHost);

  cudaFree(A_gpu);
  cudaFree(r_gpu);
  cudaFree(s_gpu);
  cudaFree(p_gpu);
  cudaFree(q_gpu);
}

template<typename T>
void doitgenCuda(int nr, int nq, int np,
        T* A,
        T* C4,
        T* sum_outputFromGpu)
{
  T* AGpu;
  T* C4Gpu;
  T* sumGpu;

  cudaMalloc(&AGpu, nr * nq * np * sizeof(T));
  cudaMalloc(&C4Gpu, np * np * sizeof(T));
  cudaMalloc(&sumGpu, nr * nq * np * sizeof(T));

  cudaMemcpy(AGpu, A, nr * nq * np * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(C4Gpu, C4, np * np * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(sumGpu, sum_outputFromGpu, nr * nq * np * sizeof(T), cudaMemcpyHostToDevice);

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((unsigned int)ceil( ((float)np) / ((float)block.x) ), (unsigned int)ceil( ((float)nr) / ((float)block.y) ));

  /* Start timer. */
  polybench_start_instruments;  

  for (int r = 0; r < nr; r++)
  {
    doitgen_kernel1 <<<grid, block>>> (nr, nq, np, sumGpu, AGpu, C4Gpu, r);
    cudaThreadSynchronize();
    doitgen_kernel2 <<<grid, block>>> (nr, nq, np, sumGpu, AGpu, C4Gpu, r);
    cudaThreadSynchronize();
  }

  /* Stop and print timer. */
  printf("GPU Time in seconds:\n");
    polybench_stop_instruments;
  polybench_print_instruments;
    
  cudaMemcpy(sum_outputFromGpu, sumGpu, NR * nq * np * sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(AGpu);
  cudaFree(C4Gpu);
  cudaFree(sumGpu);
}

template<typename T>
void gemmCuda(int ni, int nj, int nk, T alpha, T beta, T* A, 
  T* B, T* C, T* C_outputFromGpu)
{
  T *A_gpu;
  T *B_gpu;
  T *C_gpu;

  cudaMalloc((void **)&A_gpu, sizeof(T) * ni * nk);
  cudaMalloc((void **)&B_gpu, sizeof(T) * nk * nj);
  cudaMalloc((void **)&C_gpu, sizeof(T) * ni * nj);
  
  cudaMemcpy(A_gpu, A, sizeof(T) * ni * nk, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, sizeof(T) * nk * nj, cudaMemcpyHostToDevice);
  cudaMemcpy(C_gpu, C, sizeof(T) * ni * nj, cudaMemcpyHostToDevice);
  
  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((size_t)(ceil( ((float)ni)/ ((float)block.x) )),(size_t)(ceil( ((float)nj)/ ((float)block.y) )));

  /* Start timer. */
    polybench_start_instruments;

  gemm_kernel<<< grid, block >>>(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu);
  cudaThreadSynchronize();

  /* Stop and print timer. */
  printf("GPU Time in seconds:\n");
    polybench_stop_instruments;
   polybench_print_instruments;

  cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(T) * ni * nj, cudaMemcpyDeviceToHost);    
  
  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
}

template<typename T>
void gemverCuda(int n, T alpha, T beta,
    T* A,
    T* u1,
    T* v1,
    T* u2,
    T* v2,
    T* w,
    T* w_outputFromGpu,
    T* x,
    T* y,
    T* z)
{
  T *A_gpu;
  T *x_gpu;
  T *y_gpu;
  T *z_gpu;
  T *v1_gpu;
  T *v2_gpu;
  T *u1_gpu;
  T *u2_gpu;
  T *w_gpu;

  cudaMalloc((void **)&A_gpu, sizeof(T) * n * n);
  cudaMalloc((void **)&x_gpu, sizeof(T) * n);
  cudaMalloc((void **)&y_gpu, sizeof(T) * n);
  cudaMalloc((void **)&z_gpu, sizeof(T) * n);
  cudaMalloc((void **)&w_gpu, sizeof(T) * n);
  cudaMalloc((void **)&v1_gpu, sizeof(T) * n);
  cudaMalloc((void **)&v2_gpu, sizeof(T) * n);
  cudaMalloc((void **)&u1_gpu, sizeof(T) * n);
  cudaMalloc((void **)&u2_gpu, sizeof(T) * n);
  
  cudaMemcpy(A_gpu, A, sizeof(T) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(x_gpu, x, sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(y_gpu, y, sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(z_gpu, z, sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(w_gpu, w, sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(v1_gpu, v1, sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(v2_gpu, v2, sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(u1_gpu, u1, sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(u2_gpu, u2, sizeof(T) * n, cudaMemcpyHostToDevice);

  dim3 block1(DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y);
  dim3 grid1((size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X)), (size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_1_Y)));

  dim3 block2(DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y);
  dim3 grid2((size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)), 1);
  
  dim3 block3(DIM_THREAD_BLOCK_KERNEL_3_X, DIM_THREAD_BLOCK_KERNEL_3_Y);
  dim3 grid3((size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X)), 1);
  
   /* Start timer. */
    polybench_start_instruments;

  gemver_kernel1<<< grid1, block1 >>>(n, alpha, beta, A_gpu,v1_gpu,v2_gpu, u1_gpu, u2_gpu);
  cudaThreadSynchronize();
  gemver_kernel2<<< grid2, block2 >>>(n, alpha, beta, A_gpu,x_gpu,y_gpu, z_gpu);
  cudaThreadSynchronize();
  gemver_kernel3<<< grid3, block3 >>>(n, alpha, beta, A_gpu,x_gpu,w_gpu);
  cudaThreadSynchronize();

  /* Stop and print timer. */
  printf("GPU Time in seconds:\n");
    polybench_stop_instruments;
   polybench_print_instruments;

  cudaMemcpy(w_outputFromGpu, w_gpu, sizeof(T) * n, cudaMemcpyDeviceToHost);
  
  cudaFree(A_gpu);
  cudaFree(x_gpu);
  cudaFree(y_gpu);
  cudaFree(z_gpu);
  cudaFree(w_gpu);
  cudaFree(v1_gpu);
  cudaFree(v2_gpu);
  cudaFree(u1_gpu);
  cudaFree(u2_gpu);
}

template<typename T>
void gesummvCuda(int n, T alpha, T beta, T* A, T* B, 
    T* tmp, T* x, T* y,  
    T* y_outputFromGpu)
{
  T *A_gpu;
  T *B_gpu;
  T *x_gpu;
  T *y_gpu;
  T *tmp_gpu;

  cudaMalloc((void **)&A_gpu, sizeof(T) * n * n);
  cudaMalloc((void **)&B_gpu, sizeof(T) * n * n);
  cudaMalloc((void **)&x_gpu, sizeof(T) * n);
  cudaMalloc((void **)&y_gpu, sizeof(T) * n);
  cudaMalloc((void **)&tmp_gpu, sizeof(T) * n);
  
  cudaMemcpy(A_gpu, A, sizeof(T) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, sizeof(T) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(x_gpu, x, sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(y_gpu, y, sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(tmp_gpu, tmp, sizeof(T) * n, cudaMemcpyHostToDevice);

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((unsigned int)ceil( ((float)N) / ((float)block.x) ), 1);

  /* Start timer. */
    polybench_start_instruments;

  gesummv_kernel<<< grid, block>>>(n, alpha, beta, A_gpu, B_gpu, tmp_gpu, x_gpu, y_gpu);
  cudaThreadSynchronize();

  /* Stop and print timer. */
  printf("GPU Time in seconds:\n");
    polybench_stop_instruments;
   polybench_print_instruments;

  cudaMemcpy(y_outputFromGpu, y_gpu, sizeof(T) * n, cudaMemcpyDeviceToHost);
}

template<typename T>
void mvtCuda(int n, T* a, T* x1, T* x2, T* y_1, T* y_2, 
      T* x1_outputFromGpu, T* x2_outputFromGpu)
{
  T* a_gpu;
  T* x1_gpu;
  T* x2_gpu;
  T* y_1_gpu;
  T* y_2_gpu;

  cudaMalloc((void **)&a_gpu, sizeof(T) * n * n);
  cudaMalloc((void **)&x1_gpu, sizeof(T) * n);
  cudaMalloc((void **)&x2_gpu, sizeof(T) * n);
  cudaMalloc((void **)&y_1_gpu, sizeof(T) * n);
  cudaMalloc((void **)&y_2_gpu, sizeof(T) * n);
  cudaMemcpy(a_gpu, a, sizeof(T) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(x1_gpu, x1, sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(x2_gpu, x2, sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(y_1_gpu, y_1, sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(y_2_gpu, y_2, sizeof(T) * n, cudaMemcpyHostToDevice);
  
  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((size_t)ceil((float)N/ ((float)DIM_THREAD_BLOCK_X)), 1);

  /* Start timer. */
    polybench_start_instruments;
  
  mvt_kernel1<<<grid,block>>>(n, a_gpu,x1_gpu,y_1_gpu);
  mvt_kernel2<<<grid,block>>>(n, a_gpu,x2_gpu,y_2_gpu);
  cudaThreadSynchronize();

  /* Stop and print timer. */
  printf("GPU Time in seconds:\n");
    polybench_stop_instruments;
   polybench_print_instruments;
  
  cudaMemcpy(x1_outputFromGpu, x1_gpu, sizeof(T) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(x2_outputFromGpu, x2_gpu, sizeof(T) * n, cudaMemcpyDeviceToHost);    
  
  cudaFree(a_gpu);
  cudaFree(x1_gpu);
  cudaFree(x2_gpu);
  cudaFree(y_1_gpu);
  cudaFree(y_2_gpu);
}

template<typename T>
void syr2kCuda(int ni, int nj, T alpha, T beta, T* A, T* B, 
    T* C, T* C_outputFromGpu) 
{
  T *A_gpu;
  T *B_gpu;
  T *C_gpu;

  cudaMalloc((void **)&A_gpu, sizeof(T) * ni * nj);
  cudaMalloc((void **)&B_gpu, sizeof(T) * ni * nj);
  cudaMalloc((void **)&C_gpu, sizeof(T) * ni * ni);
  cudaMemcpy(A_gpu, A, sizeof(T) * ni * nj, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, sizeof(T) * ni * nj, cudaMemcpyHostToDevice);
  cudaMemcpy(C_gpu, C, sizeof(T) * ni * ni, cudaMemcpyHostToDevice);
  
  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((size_t)ceil( ((float)ni) / ((float)DIM_THREAD_BLOCK_X) ), (size_t)(ceil( ((float)ni) / ((float)DIM_THREAD_BLOCK_Y) )));
  
  /* Start timer. */
    polybench_start_instruments;

  syr2k_kernel<<<grid,block>>>(ni, nj, alpha, beta, A_gpu, B_gpu, C_gpu);
  cudaThreadSynchronize();

  /* Stop and print timer. */
  printf("GPU Time in seconds:\n");
    polybench_stop_instruments;
   polybench_print_instruments;
    
  cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(T) * ni * ni, cudaMemcpyDeviceToHost);

  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
}

template<typename T>
void syrkCuda(int ni, int nj, T alpha, T beta, T* A, T* C, 
    T* C_outputFromGpu)
{
  T* A_gpu;
  T* C_gpu;

  cudaMalloc((void **)&A_gpu, sizeof(T) * ni * nj);
  cudaMalloc((void **)&C_gpu, sizeof(T) * ni * ni);
  cudaMemcpy(A_gpu, A, sizeof(T) * ni * nj, cudaMemcpyHostToDevice);
  cudaMemcpy(C_gpu, C, sizeof(T) * ni * ni, cudaMemcpyHostToDevice);
  
  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((size_t)(ceil(((float)ni) / ((float)DIM_THREAD_BLOCK_X))), (size_t)ceil(((float)ni) / ((float)DIM_THREAD_BLOCK_Y)));

  /* Start timer. */
    polybench_start_instruments;

  syrk_kernel<<<grid,block>>>(ni, nj, alpha, beta, A_gpu,C_gpu);
  cudaThreadSynchronize();

  /* Stop and print timer. */
  printf("GPU Time in seconds:\n");
    polybench_stop_instruments;
   polybench_print_instruments;

  cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(T) * ni * ni, cudaMemcpyDeviceToHost);

  cudaFree(A_gpu);
  cudaFree(C_gpu);
}
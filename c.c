#include <time.h>
#include <stdio.h>
#include <immintrin.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>

#define N 1024LL
float a[N*N] __attribute__((aligned(32)));
float b[N*N] __attribute__((aligned(32)));
float c[N*N] __attribute__((aligned(32)));
float val[N*N];

#define ll long long 
ll nanotime(){
  clockid_t clk = CLOCK_MONOTONIC;
  struct timespec t; 
  int err = clock_gettime(clk,&t);
  return  t.tv_sec*1000000000+t.tv_nsec;
}


#define BLOCK 8
int const THREADS = BLOCK*BLOCK; 
void *parallel(void* P){
  size_t part = *((size_t*)P);
  size_t startRow = (part/BLOCK)*N/BLOCK;
  size_t startCol = (part%BLOCK)*N/BLOCK;
  for(size_t i = 0; i < N/BLOCK; ++i)
    for(size_t j = 0; j < N/BLOCK; ++j)
        b[(startRow+i)*N+(startCol+j)] = 0.0;


  for(size_t i = 0; i < N/BLOCK; ++i){
    for(size_t k = 0; k < N; ++k){
      for(size_t j = 0; j < N/BLOCK; ++j){
        b[(startRow+i)*N+(startCol+j)] += 
          a[(startRow+i)*N+k] * a[k*N+(startCol+j)];
      }
    }
  }
}


    
int main(){
#ifndef DEBUG
  FILE *file = fopen("mat.dat","rb");
  fread(a,4,N*N,file);
  fread(val,4,N*N,file);
  fclose(file);
#endif
  

#ifdef NAIVE
  float gflops = 0;
  for(int T = 0; T < 5; ++T){
    ll st = nanotime();
    for(size_t i = 0; i < N;++i){
      for(size_t j = 0; j < N; ++j){
        b[i*N+j] = 0;
        for(size_t k = 0; k < N; ++k){
          b[i*N+j] += a[i*N+k]*a[k*N+j];
        }
      }
    }
    ll et = nanotime();
    float gflop = N*N*N*2;
    float time = et-st;
    gflops += (gflop)/time;
    for(size_t i = 0; i < N*N; ++i)
      if(abs(b[i]-val[i]) > 1e-5){
        printf("Fail at %u, %f != %f\n",i,b[i],val[i]);
        return 0;
      }
    printf("GFLOPS : %f\n",gflop/time);
  }
  printf("Avg GFLOPS : %f\n",gflops/5);
  
#endif

#ifdef CACHE
  float gflops = 0;
  for(int T = 0; T < 5; ++T){
    for(size_t i = 0; i < N;++i)
      for(size_t j = 0; j < N; ++j)
        b[i*N+j] = 0;
    ll st = nanotime();
    for(size_t i = 0; i < N;++i){
      for(size_t k = 0; k < N; ++k){
        for(size_t j = 0; j < N; ++j){
          b[i*N+j] += a[i*N+k]*a[k*N+j];
        }
      }
    }
    ll et = nanotime();
    float gflop = N*N*N*2;
    float time = et-st;
    gflops += (gflop)/time;
    for(size_t i = 0; i < N*N; ++i)
      if(abs(b[i]-val[i]) > 1e-5){
        printf("Fail at %u, %f != %f\n",i,b[i],val[i]);
        return 0;
      }
    printf("GFLOPS : %f\n",gflop/time);
  }
  printf("Avg GFLOPS : %f\n",gflops/5);
#endif

#ifdef FAST
  float gflops = 0;
  for(int T = 0; T < 5; ++T){
    for(size_t i = 0; i < N;++i)
      for(size_t j = 0; j < N; ++j){
        b[i*N+j] = 0;
        c[i*N+j] = a[j*N+i];
      }
    ll st = nanotime();

    __m512 xm,ym,zm;
    for(size_t i = 0; i < N;++i){
      for(size_t j = 0; j < N; ++j){
        for(size_t Kstop= 0; Kstop < N; Kstop += 16){
       if (((uintptr_t)(a + (i * N + Kstop)) % 64 != 0) || ((uintptr_t)(c + (j * N + Kstop)) % 64 != 0)) {
         assert(false);
      }
        xm = _mm512_load_ps(a+(i*N+Kstop));
        ym = _mm512_load_ps(c+(j*N+Kstop));
        zm = _mm512_mul_ps(xm,ym);
        b[i*N+j] += _mm512_reduce_add_ps(zm);
        }
      }
    }

    ll et = nanotime();
    float gflop = N*N*N*2;
    float time = et-st;
    gflops += (gflop)/time;
    for(size_t i = 0; i < N*N; ++i)
      if(abs(b[i]-val[i]) > 1e-5){
        printf("Fail at %u, %f != %f\n",i,b[i],val[i]);
        return 0;
      }
    printf("GFLOPS : %f\n",gflop/time);
  }
  printf("Avg GFLOPS : %f\n",gflops/5);
#endif

#ifdef PP
  float gflops = 0;
  for(int T = 0; T < 5; ++T){
    pthread_t th[THREADS];
    ll st = nanotime();
    size_t id[THREADS];
    for(size_t t = 0; t < THREADS; ++t)
      id[t] = t;

    for(size_t t = 0; t < THREADS; ++t)
      pthread_create(&th[t], NULL,parallel,(void*)(id+t));

    for(size_t t = 0; t < THREADS; ++t)
      pthread_join(th[t], NULL);
    ll et = nanotime();
    float gflop = N*N*N*2;
    float time = et-st;
    gflops += (gflop)/time;
    for(size_t i = 0; i < N*N; ++i)
      if(abs(b[i]-val[i]) > 1e-5){
        printf("Fail at %u, %f != %f\n",i,b[i],val[i]);
        return 0;
      }
    printf("GFLOPS : %f\n",gflop/time);
  }
  printf("Avg GFLOPS : %f\n",gflops/5);

#endif 
  
#ifdef DEBUG
  for(int i = 0; i  < 16; ++i)
    a[i] = i;

  __m512 x,y,z;
  x = _mm512_load_ps(a);
  y = _mm512_load_ps(a);
  z = _mm512_mul_ps(x,y);
  _mm512_store_ps(b,z);

  for(int i = 0; i < 16; ++i)
    printf("%f ",*(b+i));
  printf("%f\n",_mm512_reduce_add_ps(z));


  printf("\n");
#endif
  return 0;

}

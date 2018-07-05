#include <cstdio>
#include <random>
#include <ctime>
#include <cmath>
#include <thread>
#include <chrono>


// ---------------------------------------------------------------------------------
int findBMU (int *mapsize,double *codebook, double *data, int dim, int from) {
// ---------------------------------------------------------------------------------
  int nnodes = mapsize[0]*mapsize[1];
  int k,dd,bmu=-1;
  float minm = 10000000;
  float m;

  for (int i=0;i<nnodes;i++) {
    m = 0;
    k = 0;
    dd = i*dim;
    while ((m < minm) && (k < dim)) {
      m = m + (data[from + k] - codebook[dd + k]) * (data[from + k] - codebook[dd + k]);
      k++;
    }

    if (m < minm) {
      minm = m;
      bmu  = i;
    }
  }

  return bmu;
}

// ---------------------------------------------------------------------------------
void updateCodebook (int mapsizeCols, int bmu, int nnodes, double radius, double *codebook, double *data, int dim, int from, double learningRate) {
// ---------------------------------------------------------------------------------
  float a,b;
  int col,row,dd;
  for (int i=0;i<nnodes;i++) {
    col = i%mapsizeCols - bmu%mapsizeCols;
    row = (int)i/mapsizeCols - (int)bmu/mapsizeCols;
    a = col*col + row*row;
    b = exp(-a/(2*radius*radius))*learningRate;
    dd = i*dim;

    for (int k=0;k<dim;k++) {
      codebook[dd + k] = codebook[dd + k]*(1-b) + data[from + k]*b;
    }

  }

}

// ---------------------------------------------------------------------------------
void train(int *mapsize, double *codebook, double *data, int dim, int from, int nnodes, double radius, double learningRate) {
// ---------------------------------------------------------------------------------
  int bmu = findBMU(mapsize,codebook,data,dim,from);
  updateCodebook (mapsize[1], bmu, nnodes, radius, codebook, data, dim, from, learningRate);
}

// ---------------------------------------------------------------------------------
extern "C" void trainSequential(double learningRate, double *data, int dim, int dlen, double *codebook, int *mapsize, double radius) {
// ---------------------------------------------------------------------------------
  int nnodes = mapsize[0]*mapsize[1];

  for (int from=0; from < dlen*dim; from = from + dim) {
    train(mapsize,codebook,data,dim,from,nnodes,radius,learningRate);
  }

}

// ---------------------------------------------------------------------------------
extern "C" void trainParallel(double learningRate, double *data, int dim, int dlen, double *codebook, int *mapsize, double radius) {
// ---------------------------------------------------------------------------------
  int nnodes = mapsize[0]*mapsize[1];
  int from;
  int ncores = 4;
  std::thread threads[ncores];
  int f = 0;
  int t = 0;

  while (f < dlen) {

  // printf("f: %d starting process %d t: %d\n",f,f%ncores,t);
  from = f*dim;
  threads[t] = std::thread(train,mapsize,codebook,data,dim,from,nnodes,radius,learningRate);

  if (t == 3) {
  // printf("f: %d waiting\n",f);
    for (auto& th : threads) {
      th.join();
    }
  }

    f++;

    if (t < 3) {
      t++;
    } else {
      t = 0;
    }

  }
}

// ---------------------------------------------------------------------------------
extern "C" void listBMUSequential(double *data, int dim, int dlen, double *codebook, int *mapsize, int *bmuList) {
// ---------------------------------------------------------------------------------
  int bmu,from;

  for (int f=0; f < dlen; f++) {
    from = f*dim;
    bmu = findBMU (mapsize, codebook, data, dim, from);
    bmuList[f] = bmu;
  }

}


// ---------------------------------------------------------------------------------
extern "C" void test(double *data, int cols, int rows) {
// ---------------------------------------------------------------------------------
  for (int i=0;i<rows;i++) {
    for (int j=0;j<cols;j++) {
      printf(" %f ",data[i*cols+j]);
    }
    printf("\n");
  }
}





// printf("nnodes: %d\n",nnodes);
// printf("mapsize: %dx%d\n",mapsize[0],mapsize[1]);

// for (int i=0;i<nnodes;i++) {
//   dd = i*dim;
//   printf("\ncodebook at node %d:\n",i);
//   for (int k=0;k<dim;k++) {
//     printf("%lf ",codebook[dd + k]);
//   }
// }
// printf("\n");

// void process(float learningRate, int dim, float *codebook, int *mapsize, float radius, float *data) {
//   float* d = new float[dim];
//   for (int i=0; i<dlen; i = i+4) {
//     d = data[i];
//     std::thread t1(train,learningRate,d,dim,codebook,mapsize,radius);
//     d = data[i+1];
//     std::thread t2(train,learningRate,d,dim,codebook,mapsize,radius);
//     d = data[i+2];
//     std::thread t3(train,learningRate,d,dim,codebook,mapsize,radius);
//     d = data[i+3];
//     std::thread t4(train,learningRate,d,dim,codebook,mapsize,radius);
//     t1.join();
//     t2.join();
//     t3.join();
//     t4.join();
//   }
// }

// int main() {
//
//   int mapsize[2]     = {200,200};
//   int nnodes         = mapsize[0]*mapsize[1];
//   int dim            = 200;
//   int dlen           = 200;
//   float learningRate = .5;
//   float radius       = .3;
//   std::chrono::steady_clock::time_point compStart,compEnd;
//   double elapsed;
//
//   printf("making codebook of size %d\n",nnodes*dim);
//   float* codebook = new float[nnodes*dim];
//   for (int i=0; i<nnodes; i++) {
//     for (int k=0; k<dim; k++) {
//       codebook[i*dim + k] = rand();
//     }
//   }
//
//   printf("making data of size %d\n",dlen*dim);
//   float** data = new float*[dlen];
//   for (int i=0; i<dlen;i++) {
//     data[i] = new float[dim];
//     for (int j=0; j<dim; j++) {
//       data[i][j] = rand();
//     }
//   }
//
//   printf("Sequential\n");
//   for (int k=0;k<5;k++) {
//
//     compStart = std::chrono::steady_clock::now();
//     float* d = new float[dim];
//     for (int i=0; i<dlen; i++) {
//       d = data[i];
//       train(learningRate,d,dim,codebook,mapsize,radius);
//     }
//     compEnd= std::chrono::steady_clock::now();
//     elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(compEnd - compStart).count();
//     printf("Computing time: %lf\n",elapsed);
//   }

  // printf("Parallel\n");
  // for (int k=0;k<5;k++) {
  //
  //   compStart = std::chrono::steady_clock::now();
  //   float* d = new float[dim];
  //   for (int i=0; i<dlen; i = i+4) {
  //     d = data[i];
  //     std::thread t1(train,learningRate,d,dim,codebook,mapsize,radius);
  //     d = data[i+1];
  //     std::thread t2(train,learningRate,d,dim,codebook,mapsize,radius);
  //     d = data[i+2];
  //     std::thread t3(train,learningRate,d,dim,codebook,mapsize,radius);
  //     d = data[i+3];
  //     std::thread t4(train,learningRate,d,dim,codebook,mapsize,radius);
  //     t1.join();
  //     t2.join();
  //     t3.join();
  //     t4.join();
  //   }
  //   compEnd= std::chrono::steady_clock::now();
  //   elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(compEnd - compStart).count();
  //   printf("Computing time: %lf\n",elapsed);
  // }


//   return 0;
// }

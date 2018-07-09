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
      m += (data[from + k] - codebook[dd + k]) * (data[from + k] - codebook[dd + k]);
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
void updateBMUlist(int *mapsize, double *codebook, double *data, int dim, int f, long *BMUlist) {
// ---------------------------------------------------------------------------------
  int from = f*dim;
  int bmu = findBMU (mapsize,codebook,data,dim,from);
  BMUlist[f] = (long)bmu;
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
extern "C" void trainParallel(double learningRate, double *data, int dim, int dlen, double *codebook, int *mapsize, double radius, int ncores) {
// ---------------------------------------------------------------------------------
  int nnodes = mapsize[0]*mapsize[1];
  int from;
  std::thread threads[ncores];
  int f = 0;
  int t = 0;

  while (f < dlen) {

    from = f*dim;
    threads[t] = std::thread(train,mapsize,codebook,data,dim,from,nnodes,radius,learningRate);

    f++;
    t++;

    if (t == ncores) {
      for (auto& th : threads) {
        th.join();
      }
      t = 0;
    }
  }

  if (t < ncores) {
    for (int tt=0; tt<t; tt++) {
      threads[tt].join();
    }
  }

}

// ---------------------------------------------------------------------------------
extern "C" void getBMUlistParallel(double *data, int dim, int dlen, double *codebook, int *mapsize, long *BMUlist, int ncores) {
// ---------------------------------------------------------------------------------
  std::thread threads[ncores];
  int t = 0;
  int f = 0;

  while (f < dlen) {

    threads[t] = std::thread(updateBMUlist, mapsize, codebook, data, dim, f, BMUlist);

    f++;
    t++;

    if (t == ncores) {
      for (auto& th : threads) {
        th.join();
      }
      t = 0;
    }

  }

  if (t < ncores) {
    for (int tt=0; tt<t; tt++) {
      threads[tt].join();
    }
  }


}

// ---------------------------------------------------------------------------------
extern "C" void getBMUlistSequential(double *data, int dim, int dlen, double *codebook, int *mapsize, long *BMUlist) {
// ---------------------------------------------------------------------------------
  for (int f=0; f < dlen; f++) {
    updateBMUlist(mapsize, codebook, data, dim, f, BMUlist);
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

// ---------------------------------------------------------------------------------
extern "C" void getUMatrix(double *codebook, int *mapsize, int dim, double *umatrix) {
// ---------------------------------------------------------------------------------
  double a;
  int counter;

  for (int i=0;i<mapsize[0];i++) {
    for (int j=0;j<mapsize[1];j++) {
      umatrix[i*mapsize[1] + j] = 0;
    }
  }

  for (int i=0;i<mapsize[0];i++) {
    for (int j=0;j<mapsize[1];j++) {
      counter = 0;
      for (int di=fmax(0,i-1);di<fmin(mapsize[0],i+2);di++) {
        for (int dj=fmax(0,j-1);dj<fmin(mapsize[1],j+2);dj++) {
          a        = 0;
          counter += 1;
          for (int k=0;k<dim;k++) {
            a += pow((codebook[i*mapsize[1] + j + k] - codebook[di*mapsize[1] + dj + k]),2);
          }
          umatrix[i*mapsize[1] + j] += sqrt(a);
        }
      }
      umatrix[i*mapsize[1] + j] /= counter-1;
      // printf("%d ",counter);
    }
  }

}




// // ---------------------------------------------------------------------------------
// int main() {
// // ---------------------------------------------------------------------------------
//   int mapsize[2];
//   int dim=100;
//   mapsize[0] = 5;
//   mapsize[1] = 10;
//   int nnodes = mapsize[0]*mapsize[1];
//   double * codebook = new double[nnodes*dim];
//   double * umatrix = new double[nnodes];
//   for (int i=0;i<mapsize[1];i++) {
//     for (int j=0;j<mapsize[0];j++) {
//       for (int k=0;k<dim;k++) {
//         codebook[i*mapsize[0] + j + k] = rand();
//       }
//     }
//   }
//
//   getUMatrix(codebook,mapsize,dim,umatrix);
//
//   for (int i=0;i<mapsize[1];i++) {
//     for (int j=0;j<mapsize[0];j++) {
//       printf("%f ",umatrix[i*mapsize[0] + j]);
//     }
//     printf("\n");
//   }
//
//
// }

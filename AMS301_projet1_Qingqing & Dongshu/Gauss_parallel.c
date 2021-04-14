#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>


#define pi 3.1416

// Parallel code by the methode Gausse-Seidel
void printASCIIpartie(double* mat, int N1, int N2, char name[])
{
  char fileName[128];
  sprintf(fileName, "matmulgauss_par_%s.txt", name);
  FILE *file;
  file = fopen(fileName, "w");
  for(int i=0; i<N1; i++){
    for(int j=0; j<N2; j++){
      fprintf(file, "%f ", mat[i*N2+j]);
    }
    fprintf(file, "\n");
  }
  fclose(file);
  
}

// Print matrix 'mat' in an ASCII file
void printASCII(double* mat, int N1, int N2, char name[], int nbTasks, int myRank, int Nloc_x)

{
  char fileName[128];
  sprintf(fileName, "matmul_gausse_3_%s.txt", name);
  for(int myRankPrint=0; myRankPrint<nbTasks; myRankPrint++){
    if (myRank == myRankPrint){
      FILE *file;
      if (myRank == 0)
      file = fopen(fileName, "w");
      else
      file = fopen(fileName, "a");
      if (myRank == 0){
        for (int j=0; j<N2+2; j++)
        fprintf(file, "%f ", mat[0*(N2+2)+j]);
        fprintf(file, "\n");
        for(int i=1; i<=Nloc_x; i++){
           for(int j=0; j<N2+2; j++){
           fprintf(file, "%f ", mat[i*(N2+2)+j]);
           }
        fprintf(file, "\n");
        }
        }
      if (myRank == nbTasks-1){
  
        for(int i=1; i<=Nloc_x; i++){
           for(int j=0; j<N2+2; j++){
           fprintf(file, "%f ", mat[i*(N2+2)+j]);
           }
          fprintf(file, "\n");
        }  
        for (int j=0; j<N2+2; j++)
        fprintf(file, "%f ", mat[(Nloc_x+1)*(N2+2)+j]);   
       
        }
        fclose(file);
  }
  MPI_Barrier(MPI_COMM_WORLD);
   }
}


int main(int argc, char* argv[])
{ // Initialisation of MPI
    MPI_Init(&argc, &argv);
    int nbTasks;
    int myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &nbTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  
  

  
  // Paramètres 
  double a = 10.0;
  double b = 10.0;
  double alpha = 0.2;
  int Nx = 1000;
  int Ny = 1000;
  double U0 = 100.0;
  double delx = a/(Nx+1);
  double delx2 = delx*delx;
  double dely = b/(Ny+1);
  double dely2 = dely*dely;
  int Na = Nx*Ny;
  double del2 = delx*delx+dely*dely;
  
   //  Compute local parameters
    int n_start_x = myRank * ceil(Nx/nbTasks)+1;
    int n_start_y = myRank * ceil(Ny/nbTasks)+1;
  
    int n_end_x = (myRank+1) * ceil(Nx/nbTasks) ;
    int n_end_y = (myRank+1) * ceil(Ny/nbTasks);
   
    n_end_x = (n_end_x <=Nx) ? n_end_x : Nx;
    n_end_y = (n_end_y <=Ny) ? n_end_y : Ny;

  
    int Nloc_x = n_end_x - n_start_x + 1;
    int Nloc_y = n_end_y - n_start_y + 1;

  // Memoru allocation + Initialization of the matrices
  double* u = malloc((Nloc_x+2)*(Ny+2) * sizeof(double));


  
  // initiation de matrice u
 for (int i =0; i<=Nloc_x+1; i++){
	u[i*(Ny+2)+0] = U0;
	u[i*(Ny+2)+(Ny+1)] = U0;
}

  for (int i = 0; i<=Nloc_x+1; i++){
      //int nGlo = n_start-1+nLoc;
	for (int j = 1; j<=Ny; j++){
	   u[i*(Ny+2)+j] = 0;
}
} 
  // Parameters of the communication
  
  // Algorithme séquentiel avec Gauss-Seidel
  
    double err0 = 1000;
    double errk = 1000;
    double res=2 ;
    double errsum;
    double start = MPI_Wtime();
  
  do{
   double f;

   
 //Phase de communication
   MPI_Request reqSend1, reqRecv1;
   if(myRank>0){
    
    MPI_Irecv(&u[0*(Ny+2)+1], Ny, MPI_DOUBLE, myRank-1, 123, MPI_COMM_WORLD, &reqRecv1);
    
    MPI_Isend(&u[1*(Ny+2)+1], Ny, MPI_DOUBLE, myRank-1, 123, MPI_COMM_WORLD, &reqSend1);
    
   }
   else{ 
   for (int j =0; j<=Ny+1; j++)
   u[0*(Ny+2)+j] = U0*(1 + alpha*(1+cos(2*pi*(dely*j-b/2)/b)));
   }
   
   
   MPI_Request reqSend2, reqRecv2; 
   if(myRank<nbTasks-1){
     
    MPI_Irecv(&u[(Nloc_x+1)*(Ny+2)+1], Ny, MPI_DOUBLE, myRank+1, 123, MPI_COMM_WORLD, &reqRecv2);
    
    MPI_Isend(&u[Nloc_x*(Ny+2)+1], Ny, MPI_DOUBLE, myRank+1, 123, MPI_COMM_WORLD, &reqSend2);
     
   }
   else{
   for (int j =0; j<=Ny+1; j++)
   u[(Nloc_x+1)*(Nx+2)+j] = U0;
   }
   
   if(myRank>0)
   MPI_Wait(&reqRecv1, MPI_STATUS_IGNORE);
   if(myRank<nbTasks-1)  
   MPI_Wait(&reqRecv2, MPI_STATUS_IGNORE);
   
// Calculation of red points 
   for(int i=1; i<=Nloc_x;i++){
    for(int j=1; j<Ny+1; j++){
     f = 0;    
     if ((i+j)% 2== 0) 
      u[i*(Ny+2)+j]=0.5*(1/del2)*(dely2*(u[(i+1)*(Ny+2)+j]+u[(i-1)*(Ny+2)+j])+delx2*(u[i*(Ny+2)+j+1]+u[i*(Ny+2)+j-1])-f*delx2*dely2);
      }
      }
      
  //Phase de communication
  // MPI_Request reqSend, reqRecv;
   MPI_Request reqSend3, reqRecv3;
   if(myRank>0){
    
    MPI_Irecv(&u[0*(Ny+2)+1], Ny, MPI_DOUBLE, myRank-1, 123, MPI_COMM_WORLD, &reqRecv3);
    
    
    MPI_Isend(&u[1*(Ny+2)+1], Ny, MPI_DOUBLE, myRank-1, 123, MPI_COMM_WORLD, &reqSend3);
    
 
   }
   else{ 
   for (int j =0; j<=Ny+1; j++)
   u[0*(Ny+2)+j] = U0*(1 + alpha*(1+cos(2*pi*(dely*j-b/2)/b)));
   }
   
   MPI_Request reqSend4, reqRecv4; 
   if(myRank<nbTasks-1){
     
    MPI_Irecv(&u[(Nloc_x+1)*(Ny+2)+1], Ny, MPI_DOUBLE, myRank+1, 123, MPI_COMM_WORLD, &reqRecv4);
    
    MPI_Isend(&u[Nloc_x*(Ny+2)+1], Ny, MPI_DOUBLE, myRank+1, 123, MPI_COMM_WORLD, &reqSend4);
    
    
   }
   else{
   for (int j =0; j<=Ny+1; j++)
   u[(Nloc_x+1)*(Nx+2)+j] = U0;
   }
   
   if(myRank>0)
   MPI_Wait(&reqRecv3, MPI_STATUS_IGNORE);
   if(myRank<nbTasks-1)  
   MPI_Wait(&reqRecv4, MPI_STATUS_IGNORE);      
      
      
      
      
      
// Calculation of black points
   for(int i=1; i<=Nloc_x;i++){
    for(int j=1; j<Ny+1; j++){
       f = 0;
      if ((i+j)% 2!= 0)
        u[i*(Ny+2)+j]=0.5*(1/del2)*(dely2*(u[(i+1)*(Ny+2)+j]+u[(i-1)*(Ny+2)+j])+delx2*(u[i*(Ny+2)+j+1]+u[i*(Ny+2)+j-1])-f*delx2*dely2);
      }
      }
   
     double errk2;
     
     errk2 = 0;
      
    for(int i=1; i<=Nloc_x;i++){
      double tmp=0;
     for(int j=1; j<Ny+1; j++){
       f = 0;
     
      tmp = f-((1/delx2)*(u[(i+1)*(Ny+2)+j]+u[(i-1)*(Ny+2)+j])-2*(1/delx2+1/dely2)*u[i*(Ny+2)+j]+(1/dely2)*(u[i*(Ny+2)+j+1]+u[i*(Ny+2)+j-1]));

      errk2 += tmp*tmp;
      
    }
    }
    
    MPI_Allreduce(&errk2,&errsum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    err0 = errk;
    errk = sqrt(errsum);
    
    res = fabs((err0-errk)/errk);
      
     }while (res>0.0001);

 double end = MPI_Wtime();
  printf("Runtime in core = %d\n is %f\n", myRank, end-start);


  // Print the matrices

  printASCII(u, Nx, Ny, "u", nbTasks, myRank,Nloc_x);
  
  // Memory deallocation

  free(u);

  
  //  Finalize MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  
  return 0;
}

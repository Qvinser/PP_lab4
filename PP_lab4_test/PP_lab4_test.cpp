/**
  * MPI Reduce example.
 **/

#include <mpi.h>
#include <cstdlib>
#include <stdio.h>
using namespace std;


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double value = 0.0 + rank;
    double sum = 0.0;

    int root = 0;

    MPI_Reduce(&value, &sum, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

    if (rank == root) {
        printf("Size = %d, sum = %e\n", size, sum);
    }

    MPI_Finalize();

    return 0;
}
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<chrono>
#include<vector>
#include "mpi.h"

/* Количество ячеек вдоль координат x, y, z */
#define in 20    
#define jn 20
#define kn 20

#define TAG_UP 1
#define TAG_DOWN 2

#define a 1

double Fresh(double, double, double);
double Ro(double, double, double);
void Inic();

/* Выделение памяти для 3D пространства */
double F[in + 1][jn + 1][kn + 1];

double hx, hy, hz;


/* Функция определения точного решения */
double Fresh(double x, double y, double z)
{
    double res;
    res = x + y + z;
    return res;
}

/* Функция задания правой части уравнения */
double Ro(double x, double y, double z)
{
    double d;
    d = -a * (x + y + z);
    return d;
}

/* Подпрограмма инициализации границ 3D пространства */
void Inic()
{
    int i, j, k;
    for (i = 0; i <= in; i++)
    {
        for (j = 0; j <= jn; j++)
        {
            for (k = 0; k <= kn; k++)
            {
                if ((i != 0) && (j != 0) && (k != 0) && (i != in) && (j != jn) && (k != kn))
                {
                    F[i][j][k] = 0;
                }
                else
                {
                    F[i][j][k] = Fresh(i * hx, j * hy, k * hz);
                }
            }
        }
    }
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double X, Y, Z;
    double max, N, t1, t2;
    double owx, owy, owz, c, e;
    double Fi, Fj, Fk, F1;

    int i, j, k, mi, mj, mk;
    int R, fl, fl1, fl2;
    int it, f;
    long int osdt;


    it = 0;
    X = 2.0;
    Y = 2.0;
    Z = 2.0;
    e = 10e-5;

    /* Размеры шагов */
    hx = X / in;
    hy = Y / jn;
    hz = Z / kn;

    owx = pow(hx, 2);
    owy = pow(hy, 2);
    owz = pow(hz, 2);

    c = 2 / owx + 2 / owy + 2 / owz + a;

    auto start = std::chrono::steady_clock::now();
    /* Инициализация границ пространства */
    Inic();
    int root = 0;
    int plane_size = (jn + 1) * (kn + 1);

    int send_rank;
    int recv_rank;
    // MPI_Status status;

    int start_layer = int(double(rank) / size * in);
    int end_layer = int(double(rank+1) / size * in);
    // направление "волны" передачи; size-1 - ранк самого верхнего слоя
    bool send_top = rank < size-1;

    /* Основной итерационный цикл */
    do
    {
        send_rank = send_top ? rank + 1 : rank - 1;
        recv_rank = !send_top ? rank + 1 : rank - 1;
        if (rank == root) recv_rank = rank + 1;
        if (rank == size-1) recv_rank = rank - 1;
        f = 1;
        MPI_Request send_req, recv_req;

        int send_tag = send_top ? TAG_UP : TAG_DOWN;
        int recv_tag = send_tag;
        if (rank == root) recv_tag = TAG_DOWN;
        if (rank == size-1) recv_tag = TAG_UP;

        int* recv_layer = recv_rank < rank ? &start_layer : &end_layer;
        int recv_code = MPI_Irecv(&(F[*recv_layer][0][0]), plane_size, MPI_DOUBLE,
            recv_rank, 1, MPI_COMM_WORLD, &recv_req);

        int* send_layer = send_top ? &end_layer : &start_layer;
        MPI_Isend(&(F[*send_layer][0][0]), plane_size, MPI_DOUBLE,
            send_rank, 1, MPI_COMM_WORLD, &send_req);

        // Границы, в которых вычисляем значения функции в зависимости от положения
        // в системе декартовых координат
        int i_start = send_top ? start_layer + 1 : start_layer;
        int i_end = send_top ? end_layer : end_layer - 1;
        for (i = i_start; i <= i_end; i++)
            for (j = 1; j < jn; j++)
            {
                for (k = 1; k < kn; k++)
                {
                    F1 = F[i][j][k];
                    Fi = (F[i + 1][j][k] + F[i - 1][j][k]) / owx;
                    Fj = (F[i][j + 1][k] + F[i][j - 1][k]) / owy;
                    Fk = (F[i][j][k + 1] + F[i][j][k - 1]) / owz;
                    F[i][j][k] = (Fi + Fj + Fk - Ro(i * hx, j * hy, k * hz)) / c;
                    if (fabs(F[i][j][k] - F1) > e)
                        f = 0;
                }
            }
        if ((rank != root) && (rank < size-1)) send_top = !send_top;
        it++;
        MPI_Wait(&send_req, MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
    } while (f == 0);

    std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;

    printf("\n rank = %d  iter = %d E = %f  T = %f seconds\n", rank, it, e, elapsed.count());

    /* Нахождение максимального расхождения полученного приближенного решения
     * и точного решения */
    max = 0.0;
    {
        for (i = start_layer; i < end_layer; i++)
        {
            for (j = 1; j < jn; j++)
            {
                for (k = 1; k < kn; k++)
                {
                    if ((F1 = fabs(F[i][j][k] - Fresh(i * hx, j * hy, k * hz))) > max)
                    {
                        max = F1;
                        mi = i; mj = j; mk = k;
                    }
                }
            }
        }

        printf(" Max differ = %f\n in point(%d,%d,%d)\n", max, mi, mj, mk);
        printf(" F[%d][10][10] = %f\n", start_layer+2, F[start_layer+2][10][10]);
    }
    MPI_Finalize();
    return(0);

}

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


std::vector<double> recv_plane() {
    // Probe recv from any rank to get recv parameters.
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    int src_rank = status.MPI_SOURCE;
    int tag = status.MPI_TAG;
    int col_size;
    MPI_Get_count(&status, MPI_DOUBLE, &col_size);

    // Recv and print column.
    std::vector<double> column(col_size);
    MPI_Recv(&column[0], col_size, MPI_DOUBLE, src_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return column;
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
    e = 0.00001;

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

    int start_layer = int(double(rank) / size * in);
    int end_layer = int(double(rank+1) / size * in);
    // направление "волны" передачи
    bool send_top = rank < size;
    bool is_initialized = rank == root;

    /* Основной итерационный цикл */
    do
    {
        send_rank = send_top ? rank + 1 : rank - 1;
        recv_rank = !send_top ? rank + 1 : rank - 1;
        f = 1;

        MPI_Request send_req, recv_req;
        if (MPI_Irecv(&(F[start_layer][0][0]), plane_size, MPI_DOUBLE,
            recv_rank, 1, MPI_COMM_WORLD, &recv_req) == MPI_SUCCESS) {
            MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
            is_initialized = true;
        }
        if (!is_initialized) continue;

        for (i = 1; i < in; i++)
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
        // Нижний слой
        if (rank == root) {
            MPI_Isend(&(F[end_layer][0][0]), plane_size, MPI_DOUBLE,
                send_rank, 1, MPI_COMM_WORLD, &send_req);
        }
        // Верхний слой
        else if (rank == size-1) {
            MPI_Isend(&(F[start_layer][0][0]), plane_size, MPI_DOUBLE,
                send_rank, 1, MPI_COMM_WORLD, &send_req);
        }
        // Промежуточный слой
        else {
            if (send_top) {
                MPI_Isend(&(F[end_layer][0][0]), plane_size, MPI_DOUBLE,
                    send_rank, 1, MPI_COMM_WORLD, &send_req);
            }
            else {
                MPI_Isend(&(F[start_layer][0][0]), plane_size, MPI_DOUBLE,
                    send_rank, 1, MPI_COMM_WORLD, &send_req);
            }
            send_top = !send_top;
        }
        it++;
    } while (f == 0);

    std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;

    printf("\n in = %d  iter = %d E = %f  T = %f seconds\n", in, it, e, elapsed.count());

    /* Нахождение максимального расхождения полученного приближенного решения
     * и точного решения */
    max = 0.0;
    {
        for (i = 1; i <= in; i++)
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
        printf(" F[10][10][10] = %f\n", F[10][10][10]);
    }
    return(0);

}

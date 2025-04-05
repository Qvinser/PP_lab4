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
#define plane_size 441

#define TAG_UP 1
#define TAG_DOWN 2

#define a 1

double Fresh(double, double, double);
double Ro(double, double, double);
void Inic();

/* Выделение памяти для 3D пространства */
double F[in + 1][jn + 1][kn + 1];
int rank, size;

double hx, hy, hz;
int i, j, k, mi, mj, mk;
double X, Y, Z;
double max, N, t1, t2;
double owx, owy, owz, c, e;
double Fi, Fj, Fk, F1;

int R, fl, fl1, fl2;
int it, f;
long int osdt;


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

void f_count_FOblast() {
    for (j = 1; j < jn; j++)
    {
        for (k = 1; k < kn; k++)
        {
            F1 = F[i][j][k];
            Fi = (F[i + 1][j][k] + F[i - 1][j][k]) / owx;
            Fj = (F[i][j + 1][k] + F[i][j - 1][k]) / owy;
            Fk = (F[i][j][k + 1] + F[i][j][k - 1]) / owz;
            F[i][j][k] = (Fi + Fj + Fk - Ro(i * hx, j * hy, k * hz)) / c;
            if (fabs(F[i][j][k] - F1) > e) {
                f = 0;
            }
        }
    }
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    it = 0;
    X = 2.0;
    Y = 2.0;
    Z = 2.0;
    e = 1e-8;

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
    int head = size - 1;

    int send_rank, recv_rank;
    int send_tag, recv_tag;
    MPI_Request send_req, recv_req;
    int recv_code = 100, send_code = 100;
    int recv_layer, send_layer;

    int start_layer = int(double(rank) * (in+1) / size);
    // Последний слой (end_layer) не принадлежит подобласти конкретного процесса
    int end_layer = int(double(rank+1) * (in+1) / size);
    // направление "волны" передачи; size-1 - ранк самого верхнего слоя
    bool send_top = rank < head;
    bool first_run = true;

    int message_found;
    /* Основной итерационный цикл */
    do
    {
        f = 1;
        send_tag = send_top ? TAG_UP : TAG_DOWN;
        recv_tag = send_tag;
        if (rank == root) recv_tag = TAG_DOWN;
        if (rank == head) recv_tag = TAG_UP;
        recv_rank = recv_tag == TAG_UP ? rank - 1 : rank + 1;
        send_rank = send_tag == TAG_UP ? rank + 1 : rank - 1;

        recv_layer = recv_tag == TAG_UP ? start_layer-1 : end_layer;
        send_layer = send_top ? end_layer-1 : start_layer;

        //if (rank==0)printf("\n%f", F[0][1][1]);
        //if (rank == 0)printf("\n%d %d", start_layer, recv_rank);
        MPI_Iprobe(recv_rank, recv_tag, MPI_COMM_WORLD, &message_found, MPI_STATUS_IGNORE);
        // 0 - иначе
        if (message_found == 1)
            recv_code = MPI_Irecv(&(F[recv_layer][0][0]), plane_size, MPI_DOUBLE,
                recv_rank, recv_tag, MPI_COMM_WORLD, &recv_req);

        if (send_top) {
            for (i = start_layer+1; i <= end_layer-1; i++) f_count_FOblast();
        }
        else {
            for (i = end_layer-2; i >= start_layer; i--) f_count_FOblast();
        }

        send_code = MPI_Isend(&(F[send_layer][0][0]), plane_size, MPI_DOUBLE,
            send_rank, send_tag, MPI_COMM_WORLD, &send_req);

        if (true) {
            if (recv_code == MPI_SUCCESS)
                MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
            if (message_found == 1 && send_code == MPI_SUCCESS)
                MPI_Wait(&send_req, MPI_STATUS_IGNORE);
        }

        if ((rank > root) && (rank < head)) send_top = !send_top;
        it++;
        first_run = false;
    } while (f == 0 || it<5000);

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

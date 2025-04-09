#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<chrono>
#include<vector>
#include "mpi.h"

// Флаги о завершении работы
int local_finish = 0;
int global_finish = 0;
bool finish_flag = false;
MPI_Request finish_request;

/* Количество ячеек вдоль координат x, y, z */
#define x0 -1
#define y0 -1
#define z0 -1

#define in 100    
#define jn 100
#define kn 100
#define plane_size ((jn+1)*(kn+1))

#define TAG_DIRECT 1
#define TAG_BACK 2

#define a 1e+5

double Fresh(double, double, double);
double Ro(double, double, double);
void Inic();

/* Выделение памяти для 3D пространства */
double*** F;
double F_tmp[jn+1][kn+1];
int rank, size;
int start_layer, end_layer, layer_count;

double hx, hy, hz;
int i, j, k, mi, mj, mk;
double X, Y, Z;
double max, N, t1, t2;
double owx, owy, owz, c, e;
double Fi, Fj, Fk, F1;

int R, fl, fl1, fl2;
int it = 0, f = 0;
long int osdt;

double*** allocate_3d_array(int dim_x, int dim_y, int dim_z) {
    // Выделяем один блок памяти
    double* data = new double[dim_x * dim_y * dim_z];

    // Указатели на строки Y
    double*** array = new double** [dim_x];
    for (int i = 0; i < dim_x; ++i) {
        array[i] = new double* [dim_y];
        for (int j = 0; j < dim_y; ++j) {
            array[i][j] = &data[(i * dim_y + j) * dim_z];
        }
    }

    return array;
}

void deallocate_3d_array(double*** array, int dim_x, int dim_y) {
    delete[] & (array[0][0][0]); // удаляем data
    for (int i = 0; i < dim_x; ++i) {
        delete[] array[i];
    }
    delete[] array;
}

/* Функция определения точного решения */
double Fresh(double x, double y, double z)
{
    double res;
    res = (x * x + y * y + z * z);
    return res;
}

/* Функция задания правой части уравнения */
double Ro(double x, double y, double z)
{
    double d;
    d = 6 - a * (x * x + y * y + z * z);
    return d;
}

/* Подпрограмма инициализации границ 3D пространства */
void Inic()
{
    int i, j, k;
    for (i = start_layer; i <= end_layer; i++)
    {
        int i_sh = i - start_layer;
        for (j = 0; j <= jn; j++)
        {
            for (k = 0; k <= kn; k++)
            {
                if ((i != 0) && (j != 0) && (k != 0) && (i != in) && (j != jn) && (k != kn))
                {
                    F[i_sh][j][k] = 0;
                }
                else
                {
                    F[i_sh][j][k] = Fresh(x0 + i * hx, y0 + j * hy, z0 + k * hz);
                }
            }
        }
    }
}

void f_count_FOblast() {
    int i_sh = i - start_layer;
    for (j = 1; j < jn; j++)
    {
        for (k = 1; k < kn; k++)
        {
            F1 = F[i_sh][j][k];
            Fi = (F[i_sh + 1][j][k] + F[i_sh - 1][j][k]) / owx;
            Fj = (F[i_sh][j + 1][k] + F[i_sh][j - 1][k]) / owy;
            Fk = (F[i_sh][j][k + 1] + F[i_sh][j][k - 1]) / owz;
            F[i_sh][j][k] = (Fi + Fj + Fk - Ro(x0 + i * hx, y0 + j * hy, z0 + k * hz)) / c;
            if (fabs(F[i_sh][j][k] - F1) > e) {
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
    int root = 0;
    int head = size - 1;
    // Первый слой является смежной либо глобальной границей 
    start_layer = int(double(rank) * (in) / size);
    // Последний F[end_layer] слой является смежной либо глобальной границей  
    end_layer = int(double(rank + 1) * (in) / size);
    layer_count = end_layer - start_layer + 1;

    F = allocate_3d_array(in + 1, jn + 1, kn + 1);

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
    int test_flag = 0;

    MPI_Request send_req, recv_req, send_back_req;
    int recv_code = 100, send_code = 100, send_back_code = 100, reduce_code = 100;

    int message_back_found = 0, last_message_it = it;
    /* Основной итерационный цикл */
    do
    {
        if (f == 1 && local_finish == 0) local_finish = 1;
        MPI_Iallreduce(&local_finish, &global_finish,
            1, MPI_INT, MPI_LAND, MPI_COMM_WORLD, &finish_request);

        f = 1;
        // Обратное распространение
        if (rank < head) {
            MPI_Iprobe(rank + 1, TAG_BACK, MPI_COMM_WORLD, &message_back_found, MPI_STATUS_IGNORE);
            if (message_back_found == 1) {
                recv_code = MPI_Irecv(&(F[layer_count-1][0][0]), plane_size, MPI_DOUBLE,
                    rank + 1, TAG_BACK, MPI_COMM_WORLD, &recv_req);
                last_message_it = it;
            }
        }
        // Прямое распространение
        if (rank != root) {
            MPI_Recv(&(F[0][0][0]), plane_size, MPI_DOUBLE,
                rank - 1, TAG_DIRECT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // Вычисляем границу для обратного распространения
        i = start_layer + 1;
        f_count_FOblast();
        if (rank != root) {
            send_back_code = MPI_Isend(&(F[1][0][0]), plane_size, MPI_DOUBLE,
                rank - 1, TAG_BACK, MPI_COMM_WORLD, &send_back_req);
        }
        // Вычисляем остальное подпространство
        for (i = start_layer + 2; i < end_layer; i++) f_count_FOblast();

        // Прямое распространение
        if (rank < head) {
            send_code = MPI_Isend(&(F[layer_count-2][0][0]), plane_size, MPI_DOUBLE,
                rank + 1, TAG_DIRECT, MPI_COMM_WORLD, &send_req);
            //printf(" F[%d][1][1] = %f\n", end_layer-1, F[end_layer - 1 - start_layer][1][1]);
            //printf(" end_layer-1-start_layer = %d \n layer_count - 2 = %d", end_layer - 1 - start_layer, layer_count - 2);
        }

        if (message_back_found == 1 && recv_code == MPI_SUCCESS)
            MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
        if (send_code == MPI_SUCCESS)
            MPI_Wait(&send_req, MPI_STATUS_IGNORE);
        if (send_back_code == MPI_SUCCESS)
            MPI_Wait(&send_back_req, MPI_STATUS_IGNORE);
        //printf(" %f %f %f %f\n", F[0][1][1], F[1][1][1], F[layer_count-1][1][1], F[layer_count-2][1][1]);
        //printf("\n %d", it);
        it++;

        int complete_flag = 0;
        //MPI_Test(&finish_request, &complete_flag, MPI_STATUS_IGNORE);
        //MPI_Wait(&finish_request, MPI_STATUS_IGNORE);
    } while (!global_finish );

    std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;

    printf("\n rank = %d  iter = %d E = %f  T = %f seconds\n", rank, it, e, elapsed.count());

    /* Нахождение максимального расхождения полученного приближенного решения
     * и точного решения */
    double F2 = 0.0;
    max = 0.0;
    {
        for (i = start_layer+1; i < end_layer; i++)
        {
            for (j = 1; j < jn; j++)
            {
                for (k = 1; k < kn; k++)
                {
                    if ((F1 = fabs(F[i - start_layer][j][k]
                        - Fresh(x0 + i * hx, y0 + j * hy, z0 + k * hz))) > max)
                    {
                        F2 = F[i - start_layer][j][k];
                        max = F1;
                        mi = i; mj = j; mk = k;
                    }
                }
            }
        }

        printf(" Max differ = %f\n in point(%d,%d,%d) = %f\n", max, mi, mj, mk, F2);
        printf(" Range from %d to %d\n", start_layer, end_layer);
        //printf(" F[%d][10][10] = %f\n", start_layer+2, F[start_layer+2][10][10]);
    }
    deallocate_3d_array(F, in + 1, jn + 1);
    MPI_Finalize();
    return(0);

}
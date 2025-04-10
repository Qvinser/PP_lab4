﻿#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<chrono>

#define x0 -1
#define y0 -1
#define z0 -1

/* Количество ячеек вдоль координат x, y, z */
#define in 50
#define jn 50
#define kn 50

#define a 1e+5

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
                    F[i][j][k] = Fresh(x0 + i * hx, y0 + j * hy, z0 + k * hz);
                }
            }
        }
    }
}


int main()
{
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

    /* Основной итерационный цикл */
    do
    {
        f = 1;
        for (i = 1; i < in; i++)
            for (j = 1; j < jn; j++)
            {
                for (k = 1; k < kn; k++)
                {
                    F1 = F[i][j][k];
                    Fi = (F[i + 1][j][k] + F[i - 1][j][k]) / owx;
                    Fj = (F[i][j + 1][k] + F[i][j - 1][k]) / owy;
                    Fk = (F[i][j][k + 1] + F[i][j][k - 1]) / owz;
                    F[i][j][k] = (Fi + Fj + Fk - Ro(x0 + i * hx, y0 + j * hy, z0 + k * hz)) / c;
                    if (fabs(F[i][j][k] - F1) > e)
                        f = 0;
                }
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
                    if ((F1 = fabs(F[i][j][k] - Fresh(x0 + i * hx, y0 + j * hy, z0 + k * hz))) > max)
                    {
                        max = F1;
                        mi = i; mj = j; mk = k;
                    }
                }
            }
        }

        printf(" Max differ = %f\n in point(%d,%d,%d)\n", max, mi, mj, mk);
    }
    return(0);

}

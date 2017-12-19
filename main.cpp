#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <ctype.h>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <cblas.h>
#include "math_utils.hpp"
using namespace std;

int sum = 0;
void test2(char* p, char* q, int count1, int count2, int v)
{
    if (count1 + count2 == 0)
    {
        cout << p << endl;
        sum++;
        return;
    }

    if (v >= 0 && count1 > 0)
    {
        *q = '(';
        test2(p, q + 1, count1 - 1, count2, v + 1);
    }

    if (count2 > 0)
    {
        *q = ')';
        test2(p, q + 1, count1, count2 - 1, v - 1);
    }
}

int64_t mypow(int64_t n, int exp)
{
    int64_t value = 1;
    int i;
    for(i=1; i<=exp; ++i)
    {
        value *= n;
    }

    return value;
}

int64_t count3(int len)
{
    int64_t t=1;
    int i;
    for(i=2; i<=len; ++i)
    {
        t=t*10+mypow(10,i-1)/2;
    }

    return t;
}

int64_t test(char* x)
{
    char c = x[0];

    int len = strlen(x);
    int num = c-'0';

    if(len ==1)
    {
        if(num >= 3)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }

    int64_t t= count3(len-1);
    cout<<"count3:"<<t<<",len:"<<len-1<<endl;
    x++;
    if(num > 3)
    {
        return num * t + test(x)+ mypow(10,len-1)/2;
    }
    else if(num == 3)
    {
        return num * t + (atoi(x)+1)/2 + test(x);
    }
    else
    {
        return num * t + test(x);
    }
}


void solution(char *line)
{
    int a;
    // 在此处理单行测试数据
    sscanf(line,"%d",&a);
    // 打印处理结果
    printf("%lld\n", test(line));
}

void testVector()
{
    boost::posix_time::ptime start_cpu_;
    boost::posix_time::ptime stop_cpu_;

    start_cpu_ = boost::posix_time::microsec_clock::local_time();
    int N = 999999999;
    vector<float> v(N);
    for(int i=0; i<N; ++i)
    {
        v[i] = 0.32455F;
        //v.push_back(0.32455F);
    }
    stop_cpu_ = boost::posix_time::microsec_clock::local_time();
    cout<<"vector 分配内存耗时: "<<(stop_cpu_ - start_cpu_).total_microseconds() <<endl;

    start_cpu_ = boost::posix_time::microsec_clock::local_time();
    float n = 0;
    for(int i=0; i<N; ++i)
    {
        n = v[i];
        //n =i;
    }
    stop_cpu_ = boost::posix_time::microsec_clock::local_time();
    cout<<"vector 寻址耗时: "<<(stop_cpu_ - start_cpu_).total_microseconds() <<endl;

    cout<<n<<endl;
}

void testArray()
{
    boost::posix_time::ptime start_cpu_;
    boost::posix_time::ptime stop_cpu_;

    start_cpu_ = boost::posix_time::microsec_clock::local_time();
    const int N = 999999999;
    float* v = new float[N];
    for(int i=0; i<N; ++i)
    {
        v[i]=0.32455F;
    }
    stop_cpu_ = boost::posix_time::microsec_clock::local_time();
    cout<<"array 分配内存耗时: "<<(stop_cpu_ - start_cpu_).total_microseconds() <<endl;

    start_cpu_ = boost::posix_time::microsec_clock::local_time();
    float n = 0;
    for(int i=0; i<N; ++i)
    {
        n = v[i];
        //n =i*i;
    }
    stop_cpu_ = boost::posix_time::microsec_clock::local_time();
    cout<<"array 寻址内存耗时: "<<(stop_cpu_ - start_cpu_).total_microseconds() <<endl;

    cout<<n<<endl;
    delete[] v;
}


#define M 100
#define N (50*8*8)
#define K (20*12*12)

void testBlas()
{

    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];
    cout<<"分配内存完成！共分配："<<(M * K+N * K+M * N)*sizeof(float)/1024/1024<<" MB"<< endl;
    boost::posix_time::ptime start_cpu_;
    boost::posix_time::ptime stop_cpu_;
    start_cpu_ = boost::posix_time::microsec_clock::local_time();

    Blas::caffe_cpu_gemm(CblasNoTrans, CblasNoTrans,M, N, K, 1.F, A, B, 0.F, C);

    stop_cpu_ = boost::posix_time::microsec_clock::local_time();
    cout<<"CBlas cblas_sgemm 耗时: "<<(stop_cpu_ - start_cpu_).total_microseconds() <<endl;

////////////////////////////////////////////////////////////////////////
    start_cpu_ = boost::posix_time::microsec_clock::local_time();
    for(int m=0; m<M ; ++m)
    {
        for(int k=0; k< K ; ++k)
        {
            for(int n=0; n< N ; ++n)
            {
                C[m*n] += A[m*k] * B[k*n];
            }
        }
    }
    stop_cpu_ = boost::posix_time::microsec_clock::local_time();
    cout<<"一般矩阵计算 耗时: "<<(stop_cpu_ - start_cpu_).total_microseconds() <<endl;

}

void testMyBlas()
{
    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];
    cout<<"分配内存完成！共分配："<<(M * K+N * K+M * N)*sizeof(float)/1024/1024<<" MB"<< endl;
    boost::posix_time::ptime start_cpu_;
    boost::posix_time::ptime stop_cpu_;
    start_cpu_ = boost::posix_time::microsec_clock::local_time();

    for(int m=0; m<M ; ++m)
    {
        for(int k=0; k< K ; ++k)
        {
            for(int n=0; n< N ; ++n)
            {
                C[m*n] += A[m*k] * B[k*n];
            }
        }
    }

    stop_cpu_ = boost::posix_time::microsec_clock::local_time();
    cout<<"CBlas cblas_sgemm 耗时: "<<(stop_cpu_ - start_cpu_).total_microseconds() <<endl;

}


int main()
{
    cout << "Hello world!" << endl;
    return 0;
}

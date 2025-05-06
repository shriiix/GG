#include <iostream>
#include <cstdlib>
#include <omp.h>
using namespace std;

void parallel_max_sum(int* data, int size, int& max_val, long long& sum_val) {
    int local_max = data[0];
    long long total_sum = 0;

    #pragma omp parallel for 
    for (int i = 0; i < size; i++) {
        if (data[i] > local_max)
            local_max = data[i];
        total_sum += data[i];
    }

    max_val = local_max;
    sum_val = total_sum;
}

int main() {
    int data_size = 1000000;
    int* data = new int[data_size];

    for (int i = 0; i < data_size; i++) {
        data[i] = rand() % 100;
    }

    int max_val;
    long long sum_val;
    parallel_max_sum(data, data_size, max_val, sum_val);

    cout << "Maximum value: " << max_val << std::endl;
    cout << "Sum value: " << sum_val << std::endl;

    delete[] data;
    return 0;
}

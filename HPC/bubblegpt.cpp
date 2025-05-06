#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
using namespace std;

void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    bool sorted = false;

    // Continue iterations until no swaps are made
    while (!sorted) {
        sorted = true;

        // Odd indexed pass
        for (int i = 1; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                sorted = false;
            }
        }

        // Even indexed pass
        for (int i = 0; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                sorted = false;
            }
        }
    }
}

int main() {
    vector<int> arr(10000);
    srand(time(nullptr)); // Use time as seed for better randomness

    for (int i = 0; i < 10000; i++) {
        arr[i] = rand() % 100;
    }

    cout << "Original array: [";
    for (int i = 0; i < 10; i++) cout << arr[i] << " ";
    cout << "...]\n";

    clock_t start = clock();
    parallelBubbleSort(arr);
    clock_t end = clock();

    cout << "Sorted array: [";
    for (int i = 0; i < 10; i++) cout << arr[i] << " ";
    cout << "...]\n";

    double time_taken = double(end - start) / CLOCKS_PER_SEC;
    cout << "Execution time: " << time_taken << " seconds\n";

    return 0;
}

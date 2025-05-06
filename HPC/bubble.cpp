#include<iostream>
#include<vector>
#include<ctime>
#include<cstdlib>
using namespace std;

void bubblesort(vector<int>&arr){
    
    int n = arr.size();
    bool sorted = false;

    while(!sorted){
        sorted = true;
        
        for(int i=0; i<n-1; i+=2){
            if(arr[i]>arr[i+1]){
                swap(arr[i], arr[i+1]);
                sorted = false;
            }
        }

        for(int i= 1; i<n-1; i+=2){
            if(arr[i]>arr[i+1]){
                swap(arr[i], arr[i+1]);
                sorted = false;
            }
        }
    }
}

int main(){
    vector<int>arr(10000);
    srand(time(nullptr));
    for(int i=0; i<1000; i++){
        arr[i] = rand()%100;
    }

    cout<<"Original array is : [";
    for(int i=0; i<10; i++) cout<<arr[i]<<" ";
    cout<<"...]\n";

    clock_t start = clock();
    bubblesort(arr);
    clock_t end = clock();

    cout<<"Sorted array: [";
    for(int i=0; i<10; i++) cout<<arr[i]<<" ";
    cout<<"...]\n";

    double time_taken = double (end - start) / CLOCKS_PER_SEC;
    cout<<"Executionn time : "<<time_taken<<"seconds \n";

    return 0;
}


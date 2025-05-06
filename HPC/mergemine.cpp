#include<iostream>
#include<vector>
#include<ctime>
#include<cstdlib>
#include<omp.h>
using namespace std;


vector<int>merge(const vector<int>&left ,const vector<int>&right){
    vector<int>merged;
    int i=0, j=0;
    while(i<left.size() && j<right.size()){
        if(left[i] <= right[j])
            merged.push_back(left[i++]);
        else
            merged.push_back(right[j++]);
    }
    while (i < left.size()) merged.push_back(left[i++]);
    while (j < right.size()) merged.push_back(right[j++]);

    return merged;
}

vector<int>mergeSort(const vector<int>& arr, int depth =0){

    int n = arr.size();
    if(n<=1) return arr;
    int mid = n/2;
    vector<int>left(arr.begin(),arr.begin()+mid);
    vector<int>right(arr.begin()+mid,arr.end());

    vector<int>sortedLeft, sortedRight;

    if(depth <=3){
        #pragma omp parallel sections
        {
            #pragma omp parallel sections
            {
                sortedLeft = mergeSort(left, depth+1);
            }
            #pragma omp parallel sections
            {
                sortedRight = mergeSort(right, depth+1);
            }
        }

    }else{
        sortedLeft = mergeSort(left, depth+1);
        sortedRight = mergeSort(right, depth+1);
    }

    return merge(sortedLeft, sortedRight);


}





int main(){
    vector<int>arr(10000);
    srand(time(nullptr));
    for(int i=0; i<10000; i++){
        arr[i] = rand() % 100;
    }
    cout<<"Original array: [\n";
    for(int i =0; i<10; i++) cout<<arr[i]<< " ";
    clock_t start = clock();
    vector<int> sortedArr = mergeSort(arr);
    clock_t end = clock();

    cout<<"Sorted array: [\n";
    for(int i=0; i<10; i++) cout<<sortedArr[i]<<" ";
    cout<<"...]";
    double time_taken = double(end - start) / CLOCKS_PER_SEC;
    cout<<"Execution time: "<<time_taken;

}
// Code to implement DFS using OpenMP:
#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>
using namespace std;

const int MAX = 100000;
vector<int> graph[MAX];
bool visited[MAX];

void dfs(int start) {
    stack<int> s;
    s.push(start);

    while (!s.empty()) {
        int curr_node;

        #pragma omp critical
        {
            curr_node = s.top();
            s.pop();
        }

        if (!visited[curr_node]) {
            #pragma omp critical
            visited[curr_node] = true;

            cout << curr_node << " ";

            #pragma omp parallel for
            for (int i = 0; i < graph[curr_node].size(); i++) {
                int adj_node = graph[curr_node][i];
                if (!visited[adj_node]) {
                    #pragma omp critical
                    s.push(adj_node);
                }
            }
        }
    }
}

int main() {
    int n, m, start_node;
    cout << "Enter number of nodes, edges and the start node: ";
    cin >> n >> m >> start_node;

    cout << "Enter pairs of edges:\n";
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u); // For undirected graph
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        visited[i] = false;
    }

    cout << "DFS traversal:\n";
    dfs(start_node);

    return 0;
}
//output:
// Enter number of nodes, edges and the start node: 6 5 0
// Enter pairs of edges:
// 0 1
// 0 2
// 1 3
// 1 4
// 2 5
// DFS traversal:
// 0 2 5 1 4 3
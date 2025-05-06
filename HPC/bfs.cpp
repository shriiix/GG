#include <iostream>
#include <queue>
#include <vector>
#include <omp.h>
using namespace std;

class Node {
public:
    int data;
    Node* left;
    Node* right;
};

class BreadthFS {
public:
    Node* insert(Node* root, int data);  // BST insert
    void bfs(Node* root);                // Parallel BFS
};

// Insert node in BST order
Node* BreadthFS::insert(Node* root, int data) {
    if (!root) {
        Node* newNode = new Node();
        newNode->data = data;
        newNode->left = newNode->right = nullptr;
        return newNode;
    }

    if (data < root->data)
        root->left = insert(root->left, data);
    else
        root->right = insert(root->right, data);

    return root;
}

// Parallel BFS using OpenMP (thread-safe)
void BreadthFS::bfs(Node* root) {
    if (!root)
        return;

    queue<Node*> q;
    q.push(root);

    cout << "BFS traversal:\n";

    while (!q.empty()) {
        int qSize = q.size();
        vector<Node*> levelNodes;

        // Get current level nodes
        for (int i = 0; i < qSize; ++i) {
            levelNodes.push_back(q.front());
            q.pop();
        }

        vector<Node*> nextLevel;

        #pragma omp parallel for
        for (int i = 0; i < levelNodes.size(); ++i) {
            Node* curr = levelNodes[i];

            #pragma omp critical
            cout << curr->data << " ";

            if (curr->left) {
                #pragma omp critical
                nextLevel.push_back(curr->left);
            }
            if (curr->right) {
                #pragma omp critical
                nextLevel.push_back(curr->right);
            }
        }

        for (Node* child : nextLevel) {
            q.push(child);
        }
    }

    cout << endl;
}

int main() {
    Node* root = nullptr;
    BreadthFS bfsTree;
    int data;
    char ans;

    do {
        cout << "Enter data => ";
        cin >> data;
        root = bfsTree.insert(root, data);

        cout << "Do you want to insert one more node? (y/n): ";
        cin >> ans;

    } while (ans == 'y' || ans == 'Y');

    bfsTree.bfs(root);

    return 0;
}



// Output:
// Enter data => 5
// Do you want to insert one more node? (y/n) y
// Enter data => 3
// Do you want to insert one more node? (y/n) y
// Enter data => 2
// Do you want to insert one more node? (y/n) y
// Enter data => 1
// Do you want to insert one more node? (y/n) y
// Enter data => 7
// Do you want to insert one more node? (y/n) y
// Enter data => 8
// Do you want to insert one more node? (y/n) n
// 5 3 7 2 1 8
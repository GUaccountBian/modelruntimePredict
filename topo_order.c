#include <stdio.h>
#include <stdlib.h>

#define MAX_NODES 2000

int graph[MAX_NODES][MAX_NODES] = {0};
int in_degree[MAX_NODES] = {0};
int visited[MAX_NODES] = {0};
int order[MAX_NODES];
int num_nodes = 0;
FILE *output;

int is_valid_order(int *order, int num_nodes) {
    int local_indegree[MAX_NODES];
    for (int i = 0; i < num_nodes; i++)
        local_indegree[i] = in_degree[i];

    for (int i = 0; i < num_nodes; i++) {
        int v = order[i];
        if (local_indegree[v] != 0)
            return 0;
        for (int j = 0; j < num_nodes; j++) {
            if (graph[v][j])
                local_indegree[j]--;
        }
    }
    return 1;
}

void permute(int level) {
    if (level == num_nodes) {
        if (is_valid_order(order, num_nodes)) {
            for (int i = 0; i < num_nodes; i++) {
                fprintf(output, "%d ", order[i]);
            }
            fprintf(output, "\n");
        }
        return;
    }
    for (int i = 0; i < num_nodes; i++) {
        if (!visited[i]) {
            visited[i] = 1;
            order[level] = i;
            permute(level + 1);
            visited[i] = 0;
        }
    }
}

int main() {
    FILE *file = fopen("edge_list.txt", "r");
    if (!file) {
        printf("Failed to open file.\n");
        return 1;
    }

    int src, dest;
    while (fscanf(file, "%d %d", &src, &dest) != EOF) {
        graph[src][dest] = 1;
        in_degree[dest]++;
        if (src + 1 > num_nodes) num_nodes = src + 1;
        if (dest + 1 > num_nodes) num_nodes = dest + 1;
    }
    fclose(file);

    output = fopen("topo_orders.txt", "w");
    if (!output) {
        printf("Failed to open output file.\n");
        return 1;
    }

    permute(0);
    fclose(output);
    return 0;
}


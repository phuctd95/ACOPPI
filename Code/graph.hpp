#ifndef GRAPH_HPP_INCLUDED
#define GRAPH_HPP_INCLUDED
#include <string>
#include <fstream>
#include <cmath>
#include <bitset>
using namespace std;

extern void exit_with_error(string err);
extern const int MAX_NODE;

struct graph
{
    string name;
    int n, m;
    bool has_edge[MAX_NODE][MAX_NODE];
    vector<int> Adj[MAX_NODE];

    void read ()
    {
        ifstream fi (("../Data/" + name).c_str());
        if (fi.fail()) exit_with_error("Can't open graph " + name + " input file");

        fi >> n >> m;
        for (int i = 0; i < m; ++i)
        {
            int x, y; fi >> x >> y;
            Adj[x].emplace_back(y);
            Adj[y].emplace_back(x);
            has_edge[x][y] = has_edge[y][x] = true;
        }
    }
};

#endif // GRAPH_HPP_INCLUDED

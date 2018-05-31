#include <bits/stdc++.h>
using namespace std;

vector<string> graph = { "celeg.ga", "scere.ga", "dmela.ga", "hsapi.ga" };
int n;

template<typename T>
string param(string key, T value)
{
	ostringstream si;
	si << " -" << key << " " << value;
	return si.str();
}

int main ()
{
    cout.rdbuf((new ofstream("command.txt"))->rdbuf());

	for (int i = 0; i < graph.size(); ++i)
	for (int j = i + 1; j < graph.size(); ++j)
	for (int seed = 1; seed <= 10; ++seed)
	{
		string cmd = "main.exe";
		cmd += param("i1", graph[i]);
		cmd += param("i2", graph[j]);
		cmd += param("r", 200);
		cmd += param("ant", 10);
		cmd += param("seed", seed);

		cout << cmd << "\n";
	}
}
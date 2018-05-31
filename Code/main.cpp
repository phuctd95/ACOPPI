// C++ headers
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <set>
#include <cstring>
#include <ctime>
#include <queue>
#include <thread>
#include <atomic>

// Constants
const double eps = 1e-8;
const int MAX_NODE = 12000;

// Custom headers
#include "random_picker.hpp"
#include "graph.hpp"

// ------------------------------------

string version;                         // ten phien ban
ofstream flog;                          // file log
string file_name;                       // ten file log va file out

graph g1, g2;                           // 2 do thi
vector<vector<double>> similar_score;   // vector chua diem similar

int number_of_round;                    // so round kien chay
int ant_per_round;                      // so kien chay 1 round
int ant_parallel;                       // so kien chay dong thoi
int time_limit;                         // gioi han thoi gian chay
int reset_count_down;                   // so lan kq can lap de reset mui
double alpha1, beta1;                   // so mu cho T
double alpha2, beta2;                   // so mu cho T_node
double rho;                             // toc do bay hoi
double rebuild_ratio;                   // ti le nut can rebuild khi local_search
double rebuild_ratio_delta;             // luong rebuild_ratio de giam sau moi interval
int rebuild_ratio_interval;             // interval de giam rebuild_ratio
int ls_flag;                            // cach local_search
int algo_flag;                          // loai thuat toan
int output_flag;                        // flag chon output
atomic<int> seed;                               // so khoi tao ham random
double node_limit;                      // so dinh duoc chon lam tap tiem nang khi kien xay dung loi giai
vector<double> T_node;

/* output_flag :
    - bit 0 = flog
    - bit 1 = cout
    - bit 2 = fo (result file)
*/

/* ls_flag :
    - 0 : no local search
    - 1 : do local search on best ant of the round only
    - 2 : do local search on all ants
*/

/* algo_flag :
    - bit 0 = (1 : pick random nodes with priority  ; 0 : pick local best node          ) <not inplemented yet>
    - bit 1 = (1 : use pheremone                    ; 0 : no pheremone                  ) <inplemented>
    - bit 2 = (1 : use Tmid                         ; 0 : Tmin, Tmax only               ) <not inplemented yet>
*/

// ------------------------------------

inline int getbit (int x,int i) { return (x >> i) & 1; }

struct OutputStream          // synced output stream for standard output and log output
{
    void init()
    {
        if (getbit(output_flag, 0))
        {
            flog.open(("logs/" + file_name + ".log").c_str());
            flog << setprecision(5) << fixed;
        }

        if (getbit(output_flag, 1))
        {
            ios::sync_with_stdio(false);
            cout << setprecision(5) << fixed;
        }
    }

    template<typename T>
    OutputStream& operator<< (T v)
    {
        if (getbit(output_flag, 0)) flog << v;
        if (getbit(output_flag, 1)) { cout << v; cout.flush(); }
        return *this;
    }
} fout;

void exit_with_error (string err)
{
    fout << "Exit by error : " << err << "\n";
    exit(EXIT_FAILURE);
}

// ------------------------------------

struct Solution
{
    vector<int> Y;  // Y[x] = (y : exist alignment (x,y); -1 : not aligned yet)
    vector<int> X;  // X[y] = (x : exist alignment (x,y); -1 : not aligned yet)
    int EC;         // Number of matched edges
    int E1;         // Number of edges in G1
    int E2;         // Number of edges in G2
    double GNAS;    // Global network alignment score

    Solution ()
    {
        Y = vector<int> (g1.n, -1);
        X = vector<int> (g2.n, -1);
        EC = 0;
        E1 = 0;
        E2 = 0;
        GNAS = 0;
    }

    void add_alignment (int x,int y)
    {
        Y[x] = y;
        X[y] = x;

        // update EC & E1 & E2
        for (int j : g2.Adj[y])
        {
            if (X[j] != -1 && g1.has_edge[x][X[j]]) ++EC;
            if (X[j] != -1) ++E2;
        }

        for (int j : g1.Adj[x])
            if (Y[j] != -1) ++E1;

        // update GNAS
        GNAS = (EC != 0 ? (double)EC / (E1+E2-EC) : 0.0);
    }

    double alignment_score (int x,int y) const
    {
        // get EC & E1 & E2
        int nEC = 0, nE1 = 0 , nE2 = 0;

        for (int j : g2.Adj[y])
        {
            if (X[j] != -1 && g1.has_edge[x][X[j]]) ++nEC;
            if (X[j] != -1) ++nE2;
        }

        for (int j : g1.Adj[x])
            if (Y[j] != -1) ++nE1;

        // return new GNAS
        if (nEC == 0) return 0;
        return (double)nEC / (nE1+nE2-nEC);
    }

    int EC_score (int x,int y) const
    {
        // get EC
        int nEC = 0;
        for (int j : g2.Adj[y])
        if (X[j] != -1 && g1.has_edge[x][X[j]]) ++nEC;
        return nEC;
    }
};

namespace parser
{
    void set_default_parameters ()
    {
        version                 = "null";
        g1.name                 = "g1.ga";
        g2.name                 = "g2.ga";
        number_of_round         = 500;
        ant_per_round           = 100;
        ant_parallel            = 1;
        time_limit              = INT_MAX;
        reset_count_down        = 10;
        alpha1                  = 1.0;
        beta1                   = 2.0;
        alpha2                  = 1.5;
        beta2                   = 1.0;
        rho                     = 0.3;
        rebuild_ratio           = 1.0;
        rebuild_ratio_delta     = 0.1;
        rebuild_ratio_interval  = 50;
        ls_flag                 = 2;
        algo_flag               = 7;
        output_flag             = 7;
        seed                    = 0;
        node_limit              = 0.01;
    }

    void run (int argc,char *argv[])
    {
        // set parameters to default
        set_default_parameters();

        // check if changes required
        for (int i = 1; i < argc; i += 2)
        {
            if (strcmp(argv[i], "-ver"  ) == 0) version                 = argv[i + 1];
            if (strcmp(argv[i], "-i1"   ) == 0) g1.name                 = argv[i + 1];
            if (strcmp(argv[i], "-i2"   ) == 0) g2.name                 = argv[i + 1];
            if (strcmp(argv[i], "-r"    ) == 0) number_of_round         = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-ant"  ) == 0) ant_per_round           = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-core" ) == 0) ant_parallel            = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-tl"   ) == 0) time_limit              = atoi(argv[i + 1]) * CLOCKS_PER_SEC;
            if (strcmp(argv[i], "-reset") == 0) reset_count_down        = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-a1"   ) == 0) alpha1                  = atof(argv[i + 1]);
            if (strcmp(argv[i], "-b1"   ) == 0) beta1                   = atof(argv[i + 1]);
            if (strcmp(argv[i], "-a2"   ) == 0) alpha2                  = atof(argv[i + 1]);
            if (strcmp(argv[i], "-b2"   ) == 0) beta2                   = atof(argv[i + 1]);
            if (strcmp(argv[i], "-rho"  ) == 0) rho                     = atof(argv[i + 1]);
            if (strcmp(argv[i], "-rbr"  ) == 0) rebuild_ratio           = atof(argv[i + 1]);
            if (strcmp(argv[i], "-rbrd" ) == 0) rebuild_ratio_delta     = atof(argv[i + 1]);
            if (strcmp(argv[i], "-rbri" ) == 0) rebuild_ratio_interval  = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-ls"   ) == 0) ls_flag                 = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-algo" ) == 0) algo_flag               = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-out"  ) == 0) output_flag             = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-seed" ) == 0) seed                    = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-nl"   ) == 0) node_limit              = atof(argv[i + 1]);
        }

        // set file_name
        ostringstream seed_str;
        seed_str << seed;
        file_name = version + "-" + g1.name + "_" + g2.name + "." + seed_str.str();

        // preparing outputs
        fout.init();

        // input graphs
        g1.read();
        g2.read();
        if (g1.n > g2.n) exit_with_error("First graph has more nodes than the second");

        // printing current settings
        fout << "ACO for Graph Alignment Problem\n";
        fout << "Version                        : " << version                                              << "\n";
        fout << "Graph 1                        : " << g1.name                                              << "\n";
        fout << "Graph 2                        : " << g2.name                                              << "\n";
        fout << "Output file                    : " << ("logs/" + file_name + ".log")                       << "\n";
        fout << "(V1,E1)                        : " << "(" << g1.n << "," << g1.m                           << ")\n";
        fout << "(V2,E2)                        : " << "(" << g2.n << "," << g2.m                           << ")\n";

        fout << "\nParameter-settings\n";
        fout << "number_of_round                : " << number_of_round                                      << "\n";
        fout << "ant_per_round                  : " << ant_per_round                                        << "\n";
        fout << "ant_parallel                   : " << ant_parallel                                         << "\n";
        fout << "time_limit                     : " << time_limit / CLOCKS_PER_SEC                          << " (s)\n";
        fout << "reset_count_down               : " << reset_count_down                                     << "\n";
        fout << "alpha1                         : " << alpha1                                               << "\n";
        fout << "beta1                          : " << beta1                                                << "\n";
        fout << "alpha2                         : " << alpha2                                               << "\n";
        fout << "beta2                          : " << beta2                                                << "\n";
        fout << "rho                            : " << rho                                                  << "\n";
        fout << "rebuild_ratio                  : " << rebuild_ratio                                        << "\n";
        fout << "ls_flag                        : " << ls_flag                                              << "\n";
        fout << "algo_flag                      : " << algo_flag                                            << "\n";
        fout << "output_flag                    : " << output_flag                                          << "\n";
        fout << "seed                           : " << seed.load()                                          << "\n";
        fout << "node_limit                     : " << node_limit                                           << "\n";
    }
}

namespace fastNA
{
    Solution break_down (const Solution &s,int n_keep)
    {
        Solution n;
        double W[g1.n];

        for (int i = 0; i < g1.n; ++i)
            W[i] = s.alignment_score(i, s.Y[i]);

        for (int i = 0; i < n_keep; ++i)
        {
            int x = random_picker::pick_max_element(g1.n, W);
            n.add_alignment(x, s.Y[x]);
            W[x] = 0;
        }

        return n;
    }

    int best_node_to_align (const Solution &n)
    {
        vector<int> I;
        int w_best = -1;
        for (int x = 0; x < g1.n; ++x)
        if (n.Y[x] == -1)
        {
            int w = 0;
            for (int i : g1.Adj[x])
            if (n.Y[i] != -1) ++w;

            if (w > w_best)
            {
                I.clear();
                w_best = w;
            }

            if (w == w_best) I.emplace_back(x);
        }
        double W[I.size()];
        int x;
        for (int xi = 0; xi < I.size(); ++xi)
            W[xi] = g1.Adj[I[xi]].size();
        x = random_picker::pick(I.size(),W);
        return I[x];
    }

    int best_match (const Solution &n, int x)
    {
        vector<int> I;
        double w_best = -1;

        for (int y = 0; y < g2.n; ++y)
        if (n.X[y] == -1)
        {
            double w = n.alignment_score(x, y);

            if (w > w_best)
            {
                I.clear();
                w_best = w;
            }

            if (fabs(w - w_best) < eps) I.emplace_back(y);
        }

        return I[random_picker::get_rand(I.size())];
    }

    void rebuild (Solution &s, ostringstream *so = nullptr)
    {
        if (so) *so << s.GNAS;

        do {
            int n_keep = g1.n * (1 - rebuild_ratio);    // number of nodes to keep
            Solution n = break_down(s, n_keep);      // keep bests of nodes

            for (int i = n_keep; i < g1.n; ++i)
            {
                int x = best_node_to_align(n),
                    y = best_match(n, x);
                n.add_alignment(x, y);
            }

            if (s.GNAS >= n.GNAS) break;
            s = n;
            if (so) *so << " -> " << s.GNAS;
        } while (true);
    }
}

namespace checker
{
    string report (Solution s)
    {
        set<int> aligned;
        int EC = 0 , E1 = g1.m, E2 = 0;

        for (int x = 0; x < g1.n; ++x)
        {
            int y = s.Y[x];
            if (y == -1) return "Incomplete solution\n";

            if (aligned.find(y) != aligned.end()) return "Found 2 conflicting alignment\n";
            aligned.emplace(y);

            for (int i : g1.Adj[x])
                if (i < x && g2.has_edge[y][s.Y[i]]) ++EC;

            for (int j : g2.Adj[y])
                if (j < y && s.X[j] != -1) ++E2;
        }

        string rep = "";
        if (s.EC != EC) rep += "Incorrect EC value\n";
        if (fabs((double)EC / (E1+E2-EC) - s.GNAS) >= eps) rep += "Incorrect GNAS value\n";

        return rep == "" ? "ok" : rep;
    }

    void run (Solution s)
    {
        string rep = report(s);
        fout << "\nCheker :\n" << rep;

        if (rep == "ok" && output_flag == 0)
        {
            double rr = (1.0 - s.GNAS) * 100000;
            cout << "result = " << (int)rr << "\n";
        }

        if (rep == "ok" && getbit(output_flag, 2))
        {
            // Make result file
            ofstream fo (("output/" + file_name + ".out").c_str());

            fo << s.GNAS << "\n";
            fo << s.EC << "\n";
            for (int x = 0; x < g1.n; ++x) fo << x << " " << s.Y[x] << "\n";
        }
    }
}

namespace ACO
{
    double Tmax, Tmin;                      // ACO system parameter
    double Tmax_node, Tmin_node;
    vector<vector<float>> T;                // Trails

    double prev_GNAS = -1;                  // Best solution of previous round

    Solution Gbest, Ibest;                  // Best solution overall, best solution of current round
    int count_down;                         // count down till trail reset

    vector<thread> threads;                 // Threads to run
    vector<string> sout;                    // Output strings
    vector<Solution> sols;                  // Solution of each thread

    void reset_T () // reset trails to original
    {
        for (int i = 0; i < g1.n; ++i)
        for (int j = 0; j < g2.n; ++j) T[i][j] = Tmax;
    }

    void update_T ()
    {
        // evaporate
        for (int i = 0; i < g1.n; ++i)
        for (int j = 0; j < g2.n; ++j)
            T[i][j] = T[i][j] * (1 - rho) + rho * Tmin;

        // update new trail
        int cnt = 0;
        for (int x = 0; x < g1.n; ++x)
            if (Ibest.EC_score(x,Ibest.Y[x]) > 0)
                T[x][Ibest.Y[x]] += rho * (Tmax - Tmin);
            else cnt++;
    }

    void reset_T_node()
    {
        for (int i = 0; i < g1.n; ++i)
            T_node[i] = Tmax_node;
    }

    void update_T_node()
    {
        for(int i = 0; i <g1.n; ++i)
            T_node[i] = T_node[i] * (1 - rho) + rho * Tmin_node;
        for (int x = 0; x < g1.n; ++x)
            if (Ibest.EC_score(x,Ibest.Y[x]) > 0)
                T_node[x] += rho * (Tmax_node - Tmin_node);
    }

    void let_ant_run (int tid)
    {
        Solution &sol = sols[tid];
        double W[g2.n];

        ostringstream so;
        so << setprecision(5) << fixed;

        for (int node_cnt = 0; node_cnt < g1.n; ++node_cnt)
        {
            // pick node to align
            int x;
            {
                vector<int> I;
                int w_best = -1;
                for (int xi = 0; xi < g1.n; ++xi)
                if (sol.Y[xi] == -1)
                {
                    int w = 0;
                    for (int i : g1.Adj[xi])
                    if (sol.Y[i] != -1) ++w;

                    if (w > w_best)
                    {
                        I.clear();
                        w_best = w;
                    }

                    if (w == w_best) I.emplace_back(xi);
                }

                for (int xi = 0; xi < I.size(); ++xi)
                    W[xi] = pow(T_node[I[xi]], alpha2) * pow(g1.Adj[I[xi]].size(), beta2);
                x = random_picker::pick(I.size(),W);
                x = I[x];
            }

            // pick y to align with x
            int y;
            {
                // priorize nodes by trail and GNAS score when added
                set<pair<double, int>> best_nodes;
                int lim = max(1, (int)(g2.n * node_limit));

                for (int j = 0; j < g2.n; ++j)
                {
                	W[j] = 0;

                	if (sol.X[j] == -1)
                    {
                        best_nodes.emplace(pow(T[x][j], alpha1) * pow(eps + sol.alignment_score(x, j), beta1), j);
                        if (best_nodes.size() > lim) best_nodes.erase(best_nodes.begin());
                    }
                }

                for (auto node : best_nodes)
                    W[node.second] = node.first;

                y = random_picker::pick(g2.n, W);
                if (sol.X[y] != -1) exit_with_error("picked an aligned node in g2");
            }

            // add alignment (x,y)
            sol.add_alignment(x, y);
        }

        // Local search
        if (ls_flag == 2) fastNA::rebuild(sol, &so);

        // Save output
        sout[tid] = so.str();
    }

    void run ()
    {
        clock_t st = clock();
        Tmax = 1.0;
        Tmin = Tmax / g2.n;
        T = vector<vector<float>> (g1.n, vector<float> (g2.n, Tmax));
        T_node = vector<double> (g1.n, Tmax_node);
        count_down = reset_count_down;

        for (int rnd = 0; rnd < number_of_round; ++rnd)
        {
            fout << "\nRound " << rnd << " :\n";

            if (rnd % rebuild_ratio_interval == 0)
                rebuild_ratio -= rebuild_ratio_delta;

            Ibest.GNAS = -1;
            for (int ant = 0; ant < ant_per_round; ant += ant_parallel)
            {
                threads = vector<thread>(ant_parallel);
                sout = vector<string>(ant_parallel);
                sols = vector<Solution>(ant_parallel);

                for (int i = 0; i < ant_parallel && ant + i < ant_per_round; ++i)
                    threads[i] = thread(let_ant_run, i);

                for (int i = 0; i < ant_parallel && ant + i < ant_per_round; ++i)
                {
                    threads[i].join();

                    fout << sout[i] << "\n";
                    if (sols[i].GNAS > Ibest.GNAS) Ibest = sols[i];
                }
            }

            // Local search
            if (ls_flag == 1) fastNA::rebuild(Ibest);

            // Update trail
            if (getbit(algo_flag, 1) == 1)
            {
                update_T();
                update_T_node();
            }
            // Update count down
            if (fabs(Ibest.GNAS - prev_GNAS) < eps) --count_down;
            else
            {
                prev_GNAS = Ibest.GNAS;
                count_down = reset_count_down;
            }

            // reset trails if count_down reach 0
            if (count_down == 0)
            {
                reset_T();
                reset_T_node();
                count_down = reset_count_down;
            }

            // Log and report data
            if (Ibest.GNAS > Gbest.GNAS) Gbest = Ibest;
            fout << "Ibest.GNAS = " << Ibest.GNAS << "\n";
            fout << "Gbest.GNAS = " << Gbest.GNAS << "\n";
            fout << "count down = " << count_down << "\n";
            fout << "time       = " << double(clock() - st) / CLOCKS_PER_SEC << " (s)\n";

            // Check time elapsed
            if (clock() - st > time_limit)
            {
                fout << "\n - Passed time litmit -\n";
                break;
            }
        }
    }
};

int main(int argc,char *argv[])
{
    // Initialize variables
    parser::run(argc, argv);

    // Run program
    ACO::run();

    // Check result
    checker::run(ACO::Gbest);

    return EXIT_SUCCESS;
}

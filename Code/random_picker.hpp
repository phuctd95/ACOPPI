#ifndef RANDOM_PICKER_HPP_INCLUDED
#define RANDOM_PICKER_HPP_INCLUDED
#include<numeric>
#include<algorithm>
using namespace std;

extern atomic<int> seed;

namespace random_picker
{
    int get_rand (int lim) // get a random number in 0..lim-1
    {
        seed = (12345LL * seed + 67890) % (1LL << 31);
        return seed % lim;
    }

    double get_rand () // get a random number in range [0,1]
    {
        seed = (12345LL * seed + 67890) % (1LL << 31);
        return double(seed) / INT_MAX;
    }

    int pick (int sz, double w[]) // pick a random element from 0..sz-1, w[i]/sum(w) is the chance of i being picked
    {
        double r = get_rand() * accumulate(w, w + sz, 0.0);

        int i = 0;
        while (w[i] < r) r -= w[i++];
        return min(i, sz - 1);
    }
    int pick_max_element(int sz, double w[])
    {
        int i = 0;
        for (int j = 1; j < sz; ++j)
            if (w[j] > w[i])
                i = j;
        return i;
    }
};

#endif // RANDOM_PICKER_HPP_INCLUDED

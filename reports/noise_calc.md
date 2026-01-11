# Noise Calculation

## Primary Considerations

1. Only consider *Shot Noise* due the photon count. If Photon Count is $N$, Noise variance is also $N$.
2. Windowing alters the noise since different bins are weighted differently. Equivalent to altering the photon count
3. With MCX simulations, photon count is "fake" in the sense that we actually get photon survival rates. Instead of counting each detected photon as 1 photon, we need to count it as its survival rate (Between 0 to 1).
4. $\text{Photon Count} = N = \text{Survival Rate} \times \text{Window Weight}$
5. Each Measurand (Compact Stat Process) has a different noise equation. These are derived by applying "propagation of covariance". The derivations are on the appendix of [this paper](https://pubmed.ncbi.nlm.nih.gov/22612144/). Imagine the Photon Count of a specific bin is $N_i$.
   $$
   Var(\alpha) = \sum_i (\frac{\delta \alpha}{\delta N_i})^2 var(N_i)
   $$

## Binned Noise Vs. Continuous Noise

We can consider two separate setups.

1. *Binned*: We define a set of time ranges. Any photon arriving within a range is considered to have arrived at the center of that range.
2. *Continuous*: Photon arrival times are chosen as is.

The noise is always lower for the binned case. This intuitively makes sense. But we can also prove it using the Law of Total Variance. Which states that the Photon Arrival Time Variance for the the continuous case consists of two additive terms. Inner-bin variance and Inter-bin variance. The Inter-bin variance is the same as the Variance for the binned case.
$$
\text{Continuous Variance} = \text{Binned Variance} + \sum {Inner-Bin Variance}
$$

The inner-bin variance is the photon arrival time variance of all the photons arriving within a specific bin - which will always be non-zero. Therefore,

$$
\text{Continuous Variance} > \text{Binned Variance}
$$

The paper I cited above deals in Binned Variance.

Unfortunately, the concept of binned variance does not play well with windowing. Since during optimization, we can always choose a window that places all its weight on a single bin. In which case, the binned variance will drop down to zero. Since there is nothing to compute variance against. Which catapults SNR to infinity.

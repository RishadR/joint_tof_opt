# Noise Calculation

# Types of Noise
## Shot Noise
1. Shot noise due to photon count. If Photon Count : $N$, Noise variance : $N$.
2. ToF Gating alters the noise since each bin is weighted differently.
3. With MCX simulations, photon count is "fake" in the sense that we actually get photon survival rates. Instead of counting each detected photon as 1 photon, we need to count it as its survival rate (Between 0 to 1).
4. Noise Var. at bin b : $\bar{N}(b) = \sum_{j=1}^n \text{Survival Rate}_j$
5. $\text{Total Shot Noise} = \sum_b W_b^2 \bar{N} (b)$
6. Each Measurand (Compact Stat Process) has a different noise equation. These are derived by applying "propagation of covariance". (We can ignore it for this paper - might be useful for follow-ups.) The derivations are on the appendix of [this paper](https://pubmed.ncbi.nlm.nih.gov/22612144/). If Photon Count of a specific bin is $N_i$.
   $$
   Var(\alpha) = \sum_i (\frac{\delta \alpha}{\delta N_i})^2 var(N_i)
   $$

## Binned Noise Vs. Continuous Noise

We can consider two separate setups.

1. *Binned*: We define a set of time ranges. Any photon arriving within a range is considered to have arrived at the center of that range.
2. *Continuous*: Photon arrival times are chosen as is.

Intuitively, shot noise is always lower for the binned case. We can also prove it using the Law of Total Variance. Photon Arrival Time Variance for the the continuous case consists of two additive terms. Inner-bin variance and Inter-bin variance. The Inter-bin variance is the same as the Variance for the binned case.
$$
\text{Continuous Variance} = \text{Binned Variance} + \sum {Inner-Bin Variance}
$$

The inner-bin variance is the photon arrival time variance of all the photons arriving within a specific bin - which will always be non-zero. Therefore,

$$
\text{Continuous Variance} > \text{Binned Variance}
$$

The paper I cited above deals in Binned Variance.

Note: Unfortunately, the concept of binned variance does not play well with windowing. Since during optimization, we can always choose a window that places all its weight on a single bin. In which case, the binned variance will drop down to zero. Since there is nothing to compute variance against. Which catapults SNR to infinity.

## Instrument Noise

1. We consider an all-sweeping instrument noise std. dev. per bin $ \sigma$
2. $\text{Total Instrument Noise} = \sum_b W_b^2 \sigma^2$

# Adding Noise
We can either consider noise analytically (easy) or numerically add them (longer)
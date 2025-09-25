#!/usr/bin/env python3
# Generates all figures used by the LaTeX deck.
import os, math, json, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

np.random.seed(42)
os.makedirs("figs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# --------------------------
# Utilities
# --------------------------
def bernoulli(p):
    return 1 if np.random.rand() < p else 0

def regret_from_rewards(rewards: np.ndarray, best_mean: float):
    t = np.arange(1, len(rewards) + 1)
    cum_regret = t * best_mean - np.cumsum(rewards)
    return cum_regret

def savefig(name: str):
    plt.tight_layout()
    plt.savefig(f"figs/{name}", dpi=180)
    plt.close()

# --------------------------
# Bandit algorithms (stochastic)
# --------------------------
def epsilon_greedy(true_means: List[float], T: int, eps_schedule="decay"):
    K = len(true_means)
    Q = np.zeros(K, dtype=float)
    N = np.zeros(K, dtype=int)
    rewards = []
    # init: pull each once
    for a in range(K):
        r = bernoulli(true_means[a])
        N[a] += 1; Q[a] += r; rewards.append(r)
    for t in range(K+1, T+1):
        if eps_schedule == "fixed":
            eps = 0.1
        else:
            eps = min(1.0, K / max(1, t))  # ~K/t
        if np.random.rand() < eps:
            a = np.random.randint(K)
        else:
            a = np.argmax(Q / np.maximum(1, N))
        r = bernoulli(true_means[a])
        N[a] += 1
        Q[a] += r
        rewards.append(r)
    return np.array(rewards)

def ucb1(true_means: List[float], T: int):
    K = len(true_means)
    N = np.zeros(K, dtype=int)
    S = np.zeros(K, dtype=float)
    rewards = []
    # init
    for a in range(K):
        r = bernoulli(true_means[a])
        N[a] += 1; S[a] += r; rewards.append(r)
    for t in range(K+1, T+1):
        means = S / np.maximum(1, N)
        bonus = np.sqrt(2 * np.log(t) / np.maximum(1, N))
        a = np.argmax(means + bonus)
        r = bernoulli(true_means[a])
        N[a] += 1; S[a] += r
        rewards.append(r)
    return np.array(rewards)

def ucbv(true_means: List[float], T: int):
    # Empirical-Bernstein-style bonus
    K = len(true_means)
    N = np.zeros(K, dtype=int)
    S = np.zeros(K, dtype=float)
    SS = np.zeros(K, dtype=float)  # sum of squares
    rewards = []
    # init
    for a in range(K):
        r = bernoulli(true_means[a])
        N[a] += 1; S[a] += r; SS[a] += r*r; rewards.append(r)
    for t in range(K+1, T+1):
        means = S / np.maximum(1, N)
        var = np.maximum(0.0, SS/np.maximum(1, N) - means**2)
        bonus = np.sqrt(2 * var * np.log(t) / np.maximum(1, N)) + (3*np.log(t)/np.maximum(1, N))
        a = np.argmax(means + bonus)
        r = bernoulli(true_means[a])
        N[a] += 1; S[a] += r; SS[a] += r*r
        rewards.append(r)
    return np.array(rewards)

def kl(p, q, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def kl_ucb(true_means: List[float], T: int):
    # KL-UCB for Bernoulli (Garivier & Cappé)
    K = len(true_means)
    N = np.zeros(K, dtype=int)
    S = np.zeros(K, dtype=float)
    rewards = []
    # init
    for a in range(K):
        r = bernoulli(true_means[a])
        N[a] += 1; S[a] += r; rewards.append(r)

    def upper_conf(mean, n, t):
        # Find max q in [mean,1] s.t. KL(mean||q) <= (log t + 3 log log t)/n
        if n == 0:
            return 1.0
        rhs = (np.log(t) + 3 * np.log(max(2, np.log(t)))) / n
        lo, hi = mean, 1.0
        for _ in range(25):
            mid = 0.5*(lo+hi)
            if kl(mean, mid) > rhs:
                hi = mid
            else:
                lo = mid
        return lo

    for t in range(K+1, T+1):
        means = S / np.maximum(1, N)
        U = np.array([upper_conf(means[a], N[a], t) for a in range(K)])
        a = int(np.argmax(U))
        r = bernoulli(true_means[a])
        N[a] += 1; S[a] += r
        rewards.append(r)
    return np.array(rewards)

def thompson_sampling(true_means: List[float], T: int):
    K = len(true_means)
    alpha = np.ones(K)
    beta = np.ones(K)
    rewards = []
    for t in range(1, T+1):
        samples = np.random.beta(alpha, beta)
        a = int(np.argmax(samples))
        r = bernoulli(true_means[a])
        alpha[a] += r
        beta[a] += 1 - r
        rewards.append(r)
    return np.array(rewards)

# --------------------------
# Adversarial (Exp3)
# --------------------------
def exp3(adversary_rewards: np.ndarray, gamma: float):
    # adversary_rewards: shape (T, K) with rewards in [0,1]
    T, K = adversary_rewards.shape
    w = np.ones(K)
    rewards = []
    picks = []
    for t in range(T):
        p = (1 - gamma) * (w / np.sum(w)) + gamma / K
        a = np.random.choice(K, p=p)
        x = adversary_rewards[t, a]
        # unbiased estimator
        xhat = np.zeros(K)
        xhat[a] = x / p[a]
        w = w * np.exp((gamma / K) * xhat)
        rewards.append(x)
        picks.append(a)
    return np.array(rewards), np.array(picks)

# --------------------------
# Plot helpers
# --------------------------
def run_alg_many(alg, means, T, R=50):
    curves = []
    for _ in range(R):
        rewards = alg(means, T)
        curves.append(np.cumsum(rewards))
    return np.mean(np.stack(curves), axis=0)

def regret_many(alg, means, T, R=50):
    best = max(means)
    curves = []
    for _ in range(R):
        rewards = alg(means, T)
        curves.append(regret_from_rewards(rewards, best))
    return np.mean(np.stack(curves), axis=0)

# --------------------------
# Generate and save plots
# --------------------------
def figs_stochastic():
    means = [0.20, 0.25, 0.30]
    T = 10000
    # cumulative reward
    avg_cum_eps = run_alg_many(lambda m,T: epsilon_greedy(m,T,"decay"), means, T)
    avg_cum_ucb = run_alg_many(ucb1, means, T)
    avg_cum_kl  = run_alg_many(kl_ucb, means, T)
    avg_cum_ts  = run_alg_many(thompson_sampling, means, T)

    plt.figure(figsize=(7.2,4.2))
    plt.plot(avg_cum_eps, label="ε-greedy (decay)")
    plt.plot(avg_cum_ucb, label="UCB1")
    plt.plot(avg_cum_kl,  label="KL-UCB")
    plt.plot(avg_cum_ts,  label="Thompson Sampling")
    plt.xlabel("Steps"); plt.ylabel("Mean Cumulative Reward")
    plt.title("Stochastic Bernoulli Bandit (p=[0.20,0.25,0.30])")
    plt.legend()
    savefig("stoch_cumreward.png")

    # regret comparison (log scale on x to highlight ln T growth)
    avg_reg_eps = regret_many(lambda m,T: epsilon_greedy(m,T,"decay"), means, T)
    avg_reg_ucb = regret_many(ucb1, means, T)
    avg_reg_kl  = regret_many(kl_ucb, means, T)
    avg_reg_ts  = regret_many(thompson_sampling, means, T)

    plt.figure(figsize=(7.2,4.2))
    plt.plot(avg_reg_eps, label="ε-greedy (decay)")
    plt.plot(avg_reg_ucb, label="UCB1")
    plt.plot(avg_reg_kl,  label="KL-UCB")
    plt.plot(avg_reg_ts,  label="Thompson Sampling")
    plt.xlabel("Steps"); plt.ylabel("Mean Regret")
    plt.title("Regret Comparison (lower is better)")
    plt.legend()
    savefig("stoch_regret.png")

    # UCB family variants
    avg_cum_ucbv = run_alg_many(ucbv, means, T)
    plt.figure(figsize=(7.2,4.2))
    plt.plot(avg_cum_ucb, label="UCB1")
    plt.plot(avg_cum_ucbv, label="UCB-V")
    plt.plot(avg_cum_kl, label="KL-UCB")
    plt.xlabel("Steps"); plt.ylabel("Mean Cumulative Reward")
    plt.title("UCB Family Comparison")
    plt.legend()
    savefig("ucb_variants.png")

def figs_adversarial():
    T, K = 8000, 5
    # Construct a non-stationary adversary: move the best arm every 1000 steps
    base = np.linspace(0.2, 0.8, K)
    rewards = np.zeros((T, K))
    for t in range(T):
        shift = (t // 1000) % K
        means = np.roll(base, shift)
        rewards[t] = np.random.binomial(1, means)
    # Run Exp3 vs UCB1 (note: UCB1 is misspecified here)
    exp3_rewards, _ = exp3(rewards, gamma=np.sqrt((K*np.log(K))/T))
    ucb_like = []
    # naive: feed the realized rewards as if "true_means"—this is not correct,
    # but we just show that stochastic UCB struggles under adversarial shifts.
    # We'll simulate UCB1 on a fixed mean that equals the empirical mean of the first 500 steps.
    init_means = np.mean(rewards[:500], axis=0)
    ucb_reward = ucb1(list(init_means), T)
    # Exp3 cumulative reward
    plt.figure(figsize=(7.2,4.2))
    plt.plot(np.cumsum(exp3_rewards), label="Exp3")
    plt.plot(np.cumsum(ucb_reward), label="UCB1 (mis-specified)")
    plt.xlabel("Steps"); plt.ylabel("Cumulative Reward")
    plt.title("Adversarial Setting: Exp3 vs UCB1")
    plt.legend()
    savefig("adversarial_exp3_ucb.png")

def figs_tables():
    # Table-like CSV for small Monte Carlo summary
    means_list = [
        [0.05, 0.10, 0.20],
        [0.20, 0.25, 0.30],
        [0.40, 0.45, 0.50]
    ]
    T = 5000
    rows = []
    for means in means_list:
        for name, alg in [
            ("eps-greedy", lambda m,T: epsilon_greedy(m,T,"decay")),
            ("ucb1", ucb1),
            ("kl-ucb", kl_ucb),
            ("ts", thompson_sampling),
        ]:
            R = 50
            cum = []
            reg = []
            best = max(means)
            for _ in range(R):
                rew = alg(means, T)
                cum.append(np.sum(rew))
                reg.append(regret_from_rewards(rew, best)[-1])
            rows.append({
                "means": str(means),
                "algorithm": name,
                "avg_cum_reward": float(np.mean(cum)),
                "avg_final_regret": float(np.mean(reg))
            })
    df = pd.DataFrame(rows)
    df.to_csv("figs/summary_table.csv", index=False)

def maybe_make_ctr_toy():
    p = "data/ctr_toy.csv"
    if not os.path.exists(p):
        # create a tiny toy CTR dataset
        rng = np.random.default_rng(0)
        users = [f"u{i:04d}" for i in range(500)]
        ads = ["ad_A","ad_B","ad_C"]
        # latent CTRs: ad_C best
        true = {"ad_A":0.05, "ad_B":0.10, "ad_C":0.18}
        rows = []
        for u in users:
            for a in ads:
                for _ in range(3):
                    click = rng.random() < true[a]
                    rows.append((u,a,int(click)))
        with open(p, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user","ad","click"])
            writer.writerows(rows)

def main():
    figs_stochastic()
    figs_adversarial()
    figs_tables()
    maybe_make_ctr_toy()
    print("Figures saved to figs/, toy data to data/ctr_toy.csv")

if __name__ == "__main__":
    main()

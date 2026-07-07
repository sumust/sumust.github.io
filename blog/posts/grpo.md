# GRPO, or What the Critic Was For

*Part 5 of a five-part series on trust regions: [1. Rosenbrock Explorer](/blog/rosenbrock-explorer/) · [2. The Trust Region Subproblem](/blog/trust-region-subproblem/) · [3. Trust Regions, Natural Gradients, TRPO, PPO](/blog/trust_regions/) · [4. The Two KLs](/blog/two-kls/) · [5. GRPO, or What the Critic Was For](/blog/grpo/)*

GRPO is usually introduced as PPO minus the value network, which is true and explains nothing. Deleting half of an algorithm is only a good idea if that half's job no longer needs doing. So the real question is what the critic's job was, and the answer is not "estimate values." Estimating values is how the critic does its job. The job is making a noisy gradient estimator quiet. This post builds that answer from the bottom, on the same two-action bandit as [the rest of the series](/blog/trust_regions/), where every variance can be computed by hand. Then it shows that GRPO is what falls out when the setting changes enough that the job goes away.

## 1. The noise, exactly

The policy gradient theorem says

$$
\nabla_\theta J(\theta) = \mathbb{E}\!\left[ \nabla_\theta \log \pi_\theta(a) \, r(a) \right],
$$

and in practice you don't have the expectation, you have samples. Take one: play a single action $a$, observe $r(a)$, and form

$$
\hat g = \nabla_\theta \log \pi_\theta(a) \, r(a).
$$

This is an unbiased estimate of the gradient, and unbiased is the only nice thing about it.

On the bandit (one state, actions 1 and 0 paying $r_1$ and $r_0$, policy $\pi_\theta(a=1) = \sigma(\theta)$), the estimator's entire sampling distribution can be written down, because there are only two things that can happen. The score is $1 - \sigma$ for action 1 and $-\sigma$ for action 0, so

$$
\hat g = \begin{cases}
(1 - \sigma)\, r_1 & \text{with probability } \sigma, \\
-\sigma\, r_0 & \text{with probability } 1 - \sigma.
\end{cases}
$$

Put in $\sigma = 0.7$, $r_1 = 1$, $r_0 = 0$. The estimator returns $0.3$ or $0$, its mean is $0.21$, which is exactly $\sigma(1-\sigma)(r_1 - r_0)$, and its variance is what makes reinforcement learning slow. In a real problem with long episodes and big action spaces, this same construction produces estimates whose spread dwarfs their mean. The mean is right and the spread is the problem.

## 2. Subtracting for free

Here is the one identity the whole baseline industry is built on. Probabilities sum to one; differentiate that:

$$
\mathbb{E}\!\left[ \nabla_\theta \log \pi_\theta(a) \right] = \sum_a \nabla_\theta \pi_\theta(a) = \nabla_\theta \sum_a \pi_\theta(a) = \nabla_\theta 1 = 0.
$$

The score has mean zero. So for any constant $b$,

$$
\mathbb{E}\!\left[ \nabla_\theta \log \pi_\theta(a) \, (r(a) - b) \right] = \nabla_\theta J(\theta) - b \cdot 0 = \nabla_\theta J(\theta).
$$

You can shift every reward by the same amount and the gradient does not notice. The estimator has a knob that moves its variance and leaves its mean pinned. The only remaining question is where to set the knob.

## 3. The parabola

On the bandit, compute it. With baseline $b$ the two possible estimates become $(1-\sigma)(r_1 - b)$ and $-\sigma(r_0 - b)$, and the variance is a quadratic in $b$: a parabola, opening upward, with a unique minimum. Setting the derivative to zero:

$$
b^\star = (1 - \sigma)\, r_1 + \sigma\, r_0.
$$

Look at the weights. The *value* of this policy, the average reward it earns, is $V = \sigma r_1 + (1-\sigma) r_0$. The optimal baseline is the same average with the probabilities swapped: it weights each reward by the probability of the *other* action. The two agree only at $\sigma = \tfrac{1}{2}$.

And something better happens at $b^\star$. Substitute it back into the two possible estimates:

$$
(1-\sigma)(r_1 - b^\star) = \sigma(1-\sigma)(r_1 - r_0), \qquad
-\sigma(r_0 - b^\star) = \sigma(1-\sigma)(r_1 - r_0).
$$

They are equal, and equal to the true gradient. At the optimal baseline the estimator stops being random. One sample gives the exact answer, every time, with zero variance. Two outcomes and one adjustable constant make perfect cancellation possible. That is special to the bandit, not on offer in general. But the parabola is completely general, and so is the lesson that the value is not its minimum. In general the optimal baseline weights rewards by squared score, a result that goes back to Greensmith, Bartlett, and Baxter.

Put numbers on it, at $\sigma = 0.7$, $r_1 = 1$, $r_0 = 0$. No baseline: variance $0.019$. Baseline at the value, $b = V = 0.7$: variance $0.034$. Baseline at $b^\star = 0.3$: variance zero. The textbook move, subtracting the value function, nearly doubles the variance here relative to doing nothing at all. "A baseline reduces variance" is folklore with fine print: a *well-placed* baseline reduces variance, the value is usually well-placed, and nothing guarantees it.

All of this is below, live. The top panel is the exact sampling distribution of the one-sample estimate: two possible values and their probabilities, no simulation. Drag the baseline and watch the mean refuse to move; find $b^\star$ and watch the two values merge:

<iframe src="content/baseline-explorer.html" style="width:100%;height:85vh;border:none;border-radius:8px;" loading="lazy"></iframe>

## 4. Why the critic exists anyway

If the value is not even optimal, why did everyone spend a decade training value networks? Three reasons.

First, $b^\star$ is a dead end at scale: the score-squared weights that define it are exactly the kind of per-action bookkeeping you cannot afford in large action spaces, while the value is a single scalar expectation you can regress toward from data you already have. Second, in a real MDP the best constant is not constant. A state with rewards near 10 needs a different zero point than a state with rewards near 0, so $b$ becomes $b(s)$, and a function that maps states to their expected return under the current policy *is* the value function. There is no cheaper description of it. Third, once rewards arrive spread over time, the thing multiplying the score becomes a return estimate, and you get to choose how much of it comes from observed rewards (no bias, high variance) versus the value function's forecast (bias if the critic is wrong, much less variance). That dial is exactly what generalized advantage estimation is; the $\lambda$ in GAE interpolates between trusting what happened and trusting what the critic expected.

So the critic in PPO is not there to be right about values. It is there to make the gradient quiet, and being approximately right about values is the mechanism. That distinction sounds pedantic until you price it: a critic is a second network as large as the policy, a second loss, its own learning rate and its own ways of being wrong, and at language-model scale, roughly half the memory of the whole training run. That is a lot of machinery whose only purpose is variance reduction.

## 5. The setting changed under the algorithm

Now look at what reinforcement learning on a language model actually is. A prompt arrives. The policy emits one completion. A reward model scores it, once, at the end. There is no second step whose reward depends on the first, nothing to bootstrap, no horizon to propagate credit across. Whatever the tokens inside the completion are doing, the *reinforcement learning problem* being solved is: one state, one action, one terminal reward.

That is a bandit. It is the setting this series has been doing arithmetic in for four posts, and it stopped being a toy the moment fine-tuning became the application.

In a bandit, everything the critic was there for collapses into one number: $\mathbb{E}[r \mid \text{prompt}]$, the average reward this policy gets on this state. No temporal credit assignment, no GAE dial, no forecasting. And there is one more collapse, easy to miss. In classical RL you usually cannot revisit a state at will; the environment carries you off, and the critic's ability to *generalize* across states is how you get a baseline for states you'll never see twice. A prompt is not like that. You can condition on the same prompt as many times as you can afford to sample. The state is replayable, so the conditional mean does not need to be learned by a network. It can be estimated the way means are estimated: sample a few times and average.

## 6. GRPO

For each prompt, GRPO samples a group of $G$ completions, scores them, and uses the group's statistics as the baseline: each completion's advantage is its reward minus the group mean, divided by the group standard deviation, and every token in the completion inherits that number. The critic is gone. What replaces it is the observation that the critic was approximating a mean you can just take.

Notice which target the group mean estimates: $V$, the value, not $b^\star$. GRPO did not find a cleverer baseline. It found a cheaper way to compute the same baseline PPO's critic was being trained toward, in a setting where that computation is a for-loop instead of a neural network.

The rest of the machinery survives, and by now you can name each piece's job. The importance-ratio clip stays: that is the trust region proxy from [part 3](/blog/trust_regions/), bounding how fast the policy moves per update. The KL to the frozen reference stays: that is the fixed reference from [part 4](/blog/two-kls/), bounding where the policy ends up. GRPO moves it out of the reward and adds it directly to the loss, estimated per token by $\frac{\pi_{ref}}{\pi_\theta} - \log \frac{\pi_{ref}}{\pi_\theta} - 1$, which is always positive and unbiased for the KL. Both survive, doing the same jobs as before. The only piece removed is the one whose problem no longer exists.

Removing the critic has costs, and they are worth listing. The group mean is a noisy estimate of $V$, with variance that scales like $1/G$, so baseline quality is now bought with samples per prompt instead of with a network's generalization, and small groups buy less. Each sample's reward also appears inside the mean being subtracted from it, a small self-subtraction bias that shrinks with $G$. And dividing by the group standard deviation changes the estimator in kind, not just in scale: what you follow afterward is not an unbiased estimate of $\nabla J$ but, to first approximation, the gradient of a different objective, one where every prompt is reweighted by the spread of its rewards, so questions the policy finds hard pull harder on the parameters. Whether that reweighting helps or hurts is an empirical fight, not a theorem. None of this is disqualifying. All of it is the price, and the people who removed the critic knew they were paying it.

## 7. The question to carry

The series ends where it started, with a question instead of an algorithm. Every component of the PPO stack exists to solve a problem: the clip keeps updates local, the reference KL keeps the result near the starting model, the critic quiets an estimator whose variance grows with the horizon. When the setting moved to language models, the horizon collapsed, the state became replayable, and the critic's job shrank to estimating a mean you can sample directly. It still took years, and a great many parameters spent on critics, before someone acted on that. So the question worth carrying, next to the two from [part 4](/blog/two-kls/): for each piece of machinery in the method you are running, what problem was it added to solve, and does that problem still exist?

## References

1. Ronald Williams. Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. Machine Learning, 1992.
2. Evan Greensmith, Peter Bartlett, Jonathan Baxter. Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning. Journal of Machine Learning Research, 2004.
3. John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel. High-Dimensional Continuous Control Using Generalized Advantage Estimation. arXiv:1506.02438, 2015.
4. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.
5. Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, Daya Guo. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.

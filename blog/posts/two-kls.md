# The Two KLs

*Part 4 of a five-part series on trust regions: [1. Rosenbrock Explorer](/blog/rosenbrock-explorer/) · [2. The Trust Region Subproblem](/blog/trust-region-subproblem/) · [3. Trust Regions, Natural Gradients, TRPO, PPO](/blog/trust_regions/) · [4. The Two KLs](/blog/two-kls/) · [5. GRPO, or What the Critic Was For](/blog/grpo/)*

The [previous post](/blog/trust_regions/) ends with two questions to ask of any policy optimization method: what distance is being bounded, and what enforces the bound. Language model fine-tuning has a KL term too, and on the page it looks just like the trust region one. It is not the same object. It is anchored somewhere else, it is enforced somewhere else, and it is doing a different job. Both get called "the KL penalty," which is how people end up confused.

## 1. The KL that travels

The trust region KL from TRPO is $\bar D_{KL}(\pi_{\theta_{old}}, \pi_\theta)$, and the thing to notice is the subscript: $\theta_{old}$ is the policy from the *previous iterate*. Every time an update is accepted, the anchor moves to the new policy and the constraint is rebuilt around it. The region travels with you.

A traveling constraint bounds speed, not destination. Take a thousand updates, each with KL at most $\Delta$ from the last, and you can end up anywhere in policy space. Nothing about the constraint prefers one final policy over another, and that is by design: its job is to protect the *optimization*. Each update stays inside the region where the surrogate is honest, and no single step can lurch far enough to poison the data that the next step trains on.

There is a sharper way to say this. The trust region KL never appears in the objective. At any policy where the gradient is zero, a zero step satisfies the constraint at zero cost, so the constraint does not change which policies count as solutions. It shapes the path and then drops out of the answer: it is part of the algorithm, not part of the problem.

## 2. The KL that stays home

The KL in language model fine-tuning is different. The objective there is

$$
\max_\pi \; \mathbb{E}_{y \sim \pi}\left[ r(y) \right] - \beta \, D_{KL}(\pi \,\|\, \pi_{ref}),
$$

where $r$ is a learned reward model and $\pi_{ref}$ is a *fixed* reference policy, usually the model as it stood before RL began. The anchor never moves. Train for a thousand updates and the penalty still measures distance to the same frozen model it measured on step one.

A fixed anchor bounds destination, not speed. You can move as fast as your optimizer allows, but you cannot end up far from the reference without paying for every unit of distance, forever. And the job is different too: this KL protects the *model*, not the optimization. The reward model is only trustworthy near the distribution it was trained on; optimize hard enough without the leash and you find its bugs: degenerate text that scores beautifully. The penalty also has a blunter purpose: it keeps the policy a language model at all, instead of whatever distribution happens to maximize a learned scalar.

One structural detail worth noticing: the expectation in this KL runs under $\pi$, not under the reference. That direction charges an infinite price for putting probability where the reference has essentially none. The policy can sharpen inside the reference's support, but it cannot wander off it.

## 3. It moves the answer

Here is the deepest difference. Because the reference KL sits inside the objective, it changes what the optimal policy *is*. The penalized problem has a closed-form solution:

$$
\pi^\star(y) \propto \pi_{ref}(y) \, e^{r(y) / \beta}.
$$

(To see it: the objective is linear in $\pi$ plus an entropy-like term, and setting the derivative of the Lagrangian to zero gives the exponential tilt.) The optimum is the reference distribution reweighted by the reward, with $\beta$ as the dial. Send $\beta \to \infty$ and you recover the reference untouched. Send $\beta \to 0$ and all mass collapses onto the highest-reward outputs. Every fine-tuned model is somewhere on that dial.

The trust region KL does nothing of the kind. It never reweights anything; the optima of the problem are exactly where they were. One changes how you get to the optimum; the other changes the optimum.

This closed form is not a curiosity. Direct Preference Optimization starts from exactly this equation, inverts it to write the reward in terms of the policy, and thereby skips reinforcement learning entirely. A whole family of methods is built on it.

## 4. The bandit again

The two-action bandit from the [TRPO post](/blog/trust_regions/) can hold both KLs at once, and the contrast is clean. Rewards $r_1 > r_0$, policy $\pi_\theta(a=1) = \sigma(\theta)$. The unregularized optimum is $\theta \to \infty$: all mass on the better action.

Run the trust region method of the previous posts and that is still where you go. The constraint limits each step, the natural gradient marches at speed $r_1 - r_0$, and after enough updates the policy is as deterministic as you have patience for. The destination never changed; only the speed was governed.

Now add a reference policy with logit $\theta_{ref}$ and penalize $\beta \, D_{KL}(\pi_\theta \| \pi_{\theta_{ref}})$. Setting the derivative to zero gives the optimum in closed form, and in logit space it is one line:

$$
\theta^\star = \theta_{ref} + \frac{r_1 - r_0}{\beta}.
$$

A finite destination: start from the anchor, move by the reward gap over $\beta$, stop. The reward gap $r_1 - r_0$ has now shown up twice in this series in two different roles: as the natural gradient, where it sets the *speed* of every update, and here divided by $\beta$, where it sets the total *distance* the policy is allowed to travel from home. Same quantity, two different questions.

You can watch the destination move. The green curve below is plain expected return, which has no maximum short of saturation. The purple curve is the penalized objective, and its peak sits exactly at $\theta_{ref} + (r_1 - r_0)/\beta$. Turn $\beta$ up and the peak slides home to the reference; turn it down and it walks off toward saturation:

<iframe src="content/kl-penalty-explorer.html" style="width:100%;height:85vh;border:none;border-radius:8px;" loading="lazy"></iframe>

## 5. Both at once

A real RLHF run typically carries both. PPO's clip handles the per-step locality; that is the trust region idea, approximated. The $\beta$ KL to the frozen reference sits in the reward; that keeps the policy near where it started. They are not redundant, because they fail differently. Remove the clip and training destabilizes: updates tear through the region where the surrogate means anything. Remove the reference KL and training *succeeds*, in the worst way: the policy climbs the reward model right out of the region where the reward model measures anything real, and the high scores stop meaning what you wanted them to mean.

This recipe, small local updates plus a KL to a frozen reference inside the reward, is the one Ziegler and coauthors introduced for fine-tuning language models on human preferences, and the one InstructGPT carried to scale.

## 6. Summary

Two KLs, two jobs:

1. The trust region KL is anchored to the previous iterate and re-centered after every update. It bounds how fast the policy moves. It is enforced by the algorithm, appears nowhere in the objective, and vanishes from the answer: it never changes which policies are optimal. It exists to protect the optimization.
2. The reference KL is anchored to a frozen model and never moves. It bounds where the policy is allowed to end up. It sits inside the objective and reshapes the optimum itself, into the reference tilted by the reward. It exists to protect the model.

So when a paper says "we add a KL penalty," the first question is which one: where is the anchor, and does it move? Everything else about what the term is doing follows from that.

The series has one post left: [part 5](/blog/grpo/), on GRPO and what the critic was for.

## References

1. John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, Philipp Moritz. Trust Region Policy Optimization. ICML, 2015.
2. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.
3. Daniel Ziegler, Nisan Stiennon, Jeffrey Wu, Tom Brown, Alec Radford, Dario Amodei, Paul Christiano, Geoffrey Irving. Fine-Tuning Language Models from Human Preferences. arXiv:1909.08593, 2019.
4. Long Ouyang, Jeff Wu, Xu Jiang, and 17 coauthors. Training Language Models to Follow Instructions with Human Feedback. arXiv:2203.02155, 2022.
5. Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher Manning, Chelsea Finn. Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. arXiv:2305.18290, 2023.

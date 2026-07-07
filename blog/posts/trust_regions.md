# Trust Regions, Natural Gradients, TRPO, PPO

*Part 3 of a series on trust regions. The main line is [3. Trust Regions, Natural Gradients, TRPO, PPO](/blog/trust_regions/) · [4. The Two KLs](/blog/two-kls/) · [5. GRPO, or What the Critic Was For](/blog/grpo/). Two shorter companions set it up: [1. Rosenbrock Explorer](/blog/rosenbrock-explorer/) · [2. The Trust Region Subproblem](/blog/trust-region-subproblem/).*

These are notes from working through trust region methods, starting from the basics of nonlinear optimization and ending at TRPO and PPO.

The path is: iterative methods for nonlinear optimization, Newton's method and its failure modes, line search and trust regions as two ways to control the step, the trust region subproblem, why the Euclidean ball is the wrong region for policies, KL divergence as the right notion of distance, the Fisher matrix as the local form of KL, the natural gradient as steepest ascent under the Fisher metric, and TRPO as the natural gradient applied to the importance-sampled surrogate. PPO at the end as a deliberate weakening of the construction.

One example runs through the whole post: a two-action bandit with a single parameter. Every object that appears (the gradient, the Fisher matrix, the KL ball, the natural gradient, the trust region step) can be computed on it by hand, and each one will be.

## 1. The setting

Suppose you want to minimize a nonlinear $J : \mathbb{R}^n \to \mathbb{R}$. There is no closed form in general, so you build an iterative method that produces a sequence $x_0, x_1, x_2, \ldots$ converging to a local minimizer. Each step has access to local information at $x_k$: the value $J(x_k)$, the gradient $g_k = \nabla J(x_k)$, sometimes the Hessian $H_k = \nabla^2 J(x_k)$ or an approximation $B_k$.

The Taylor expansion gives a local quadratic surrogate around $x_k$:

$$
J(x_k + p) \approx m_k(p) = J(x_k) + g_k^\top p + \tfrac{1}{2} p^\top B_k p.
$$

If the surrogate were globally accurate, you would just minimize it once. It is not. The surrogate matches $J$ to second order at $x_k$ and gets less accurate as you move away. This is the central tension in nonlinear optimization: the model is useful, but only locally, and you have to decide how far to trust it.

## 2. Newton's method and its failure mode

The unconstrained minimum of $m_k$ is the Newton step:

$$
p_k^N = -B_k^{-1} g_k.
$$

If $B_k$ is positive definite, this is a real minimum, not just a critical point. Plugging in and iterating gives Newton's method, which converges quadratically near a strict minimum. Roughly, the number of correct digits doubles per step.

The failure mode is when the Newton step is taken outside its region of validity. If $J$ is highly nonlinear, the local quadratic can be a poor description of $J$ even a moderate distance from $x_k$, and the unconstrained minimum of $m_k$ might be a place where $J$ is much larger than at $x_k$. Pure Newton's method has no mechanism to notice this. It commits to the full Newton step regardless of whether the model deserves that much trust.

There are two standard ways to fix this.

## 3. Line search and trust region

**Line search.** Pick a direction first, then search along it for a step length that gives sufficient decrease in $J$. The direction is usually $-g_k$ (steepest descent) or $-B_k^{-1} g_k$ (Newton direction). The step length $\alpha_k$ is chosen by a one-dimensional procedure, most often backtracking with the Armijo condition. The structure is: direction is committed first, magnitude is negotiated.

**Trust region.** Pick a region around $x_k$ where you trust the surrogate, and solve for the best step inside that region. Both direction and magnitude come out of the constrained subproblem:

$$
\min_p \, m_k(p) \quad \text{s.t.} \quad \|p\| \le \Delta_k.
$$

The radius $\Delta_k$ is the trust radius. Small $\Delta_k$ means you do not trust the model far from $x_k$. Large $\Delta_k$ means the model has been behaving and you are willing to move further.

The radius adapts. After computing a candidate step $p_k$, you compare actual improvement against predicted:

$$
\rho_k = \frac{J(x_k) - J(x_k + p_k)}{m_k(0) - m_k(p_k)}.
$$

If $\rho_k$ is close to one, the model was honest, accept the step, possibly grow $\Delta_k$. If $\rho_k$ is poor or negative, the model lied, reject or shrink. The framework directly answers the question of how far the local model justifies moving, which is the question we wanted answered.

The reason the trust region perspective matters more than line search for what comes later: line search commits to a direction and then negotiates magnitude, but in policy optimization the direction itself is the thing we want to think about. Euclidean steepest descent gives one direction. Newton gives another. The natural gradient, which we have not derived yet, gives a third. Whether these directions are good is a question about the geometry of the parameter space, and that question is exactly what the choice of $\|\cdot\|$ in the trust region subproblem captures.

## 4. Solving the trust region subproblem

Assume $B_k$ is positive definite. If the Newton step fits inside the region, $\|B_k^{-1} g_k\| \le \Delta_k$, take it; the constraint is slack. Otherwise the solution lies on the boundary $\|p\| = \Delta_k$, and the question becomes how to find a good boundary step cheaply.

The classical answers are approximations. The **Cauchy point** minimizes the model along the steepest descent direction and truncates at the boundary; it is cheap, and it ignores curvature. **Dogleg**, due to Powell, does better by walking a two-segment path: from the zero step to the unconstrained minimizer along the steepest descent direction, then from there toward the Newton step, stopping where the path crosses the boundary. The path is a cheap sketch of the true solution curve: as $\Delta_k$ grows from zero to infinity, the exact solution of the subproblem traces a curve that leaves along $-g_k$ and bends toward the Newton step, and the dogleg approximates that curve with two line segments.

For neural-network-scale problems, none of this is available as stated, because materializing $B_k$ at all is hopeless and even a Cholesky factorization is too expensive. The standard alternative is **truncated conjugate gradient**, due to Steihaug and Toint: run conjugate gradient on the linear system $B_k p = -g_k$, which only ever needs the product of $B_k$ with a vector, and stop early if the iterate leaves the trust region or a direction of negative curvature turns up. The result lands somewhere between the Cauchy point and the Newton step, depending on how many iterations you allow. The first CG iterate is exactly the Cauchy point, and further iterations bend the step toward Newton. Hold onto this one; it comes back as the engine inside TRPO.

This section is deliberately compressed. The subproblem gets [a post of its own](/blog/trust-region-subproblem/): the exact solution, why the convergence theory rests on the Cauchy point of all things, and why the dogleg path is well-posed.

## 5. Why this geometry is wrong for policies

Here is where things switch to RL.

We have a stochastic policy $\pi_\theta(a \mid s)$ parameterized by $\theta$, and we want to maximize expected return:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)].
$$

The policy gradient theorem gives

$$
\nabla_\theta J(\theta) = \mathbb{E}\!\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, A^{\pi_\theta}(s_t, a_t) \right]
$$

up to the usual choice of advantage estimator and visitation weighting. So we have a gradient. We could plug it into a trust region method with the standard Euclidean ball $\|p\|_2 \le \Delta$ and run.

One difference from section 1 raises the stakes. In supervised learning a bad update wastes a step; the data does not care what the model did. In RL the next batch is collected by whatever policy the last update produced, so a destructive step corrupts the data for every update after it, and the damage compounds. Step control is not a refinement here. It is the problem.

The problem is that $\theta$ and $\pi_\theta$ are not the same kind of object. $\theta$ is a parameter vector. $\pi_\theta$ is a probability distribution. The Euclidean distance between two parameter vectors has no necessary relationship to how different the corresponding policies are.

Here is the smallest example that shows it, set up properly because it runs through the rest of the post. One state, two actions. Action 1 pays reward $r_1$, action 0 pays reward $r_0$, and the policy is a single logit: $\pi_\theta(a = 1) = \sigma(\theta)$. Expected return is

$$
J(\theta) = \sigma(\theta)\, r_1 + (1 - \sigma(\theta))\, r_0.
$$

A parameter step $\Delta\theta = 1$ near $\theta = 0$ moves the probability of action 1 from $0.50$ to $0.73$. The same $\Delta\theta = 1$ near $\theta = 5$ moves it from $0.993$ to $0.998$. Same Euclidean step, very different policy step. The reverse failure happens too: with a redundant parameterization, large motion in $\theta$ can correspond to negligible change in $\pi$.

The consequence is that running a Euclidean trust region method on $\theta$ produces a learning trajectory that depends on the parameterization. Rescale $\theta$, change basis, switch from logits to log-probabilities, and the dynamics change. This is bad for two reasons. First, you don't actually care about $\theta$, you care about $\pi$. Second, near saturation regions a Euclidean step does nothing to the policy, so the algorithm wastes capacity moving parameters that don't matter.

The fix is to measure the size of a step by how much the policy changes, not how much the parameters change.

## 6. KL divergence and the Fisher matrix

The standard way to measure how different two distributions are is KL divergence. For two policies parameterized by $\theta$ and $\theta + p$, expand $D_{KL}(\pi_\theta \,\|\, \pi_{\theta + p})$ in $p$ around zero. The zeroth-order term is zero (KL of a distribution with itself is zero). The first-order term is also zero, because KL is non-negative and $p = 0$ is a minimum, so the gradient there vanishes. What survives is the quadratic:

$$
D_{KL}(\pi_\theta \,\|\, \pi_{\theta + p}) = \tfrac{1}{2} p^\top F(\theta) p + O(\|p\|^3),
$$

where

$$
F(\theta) = \mathbb{E}_{s, a \sim \pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(a \mid s) \, \nabla_\theta \log \pi_\theta(a \mid s)^\top\right]
$$

is the Fisher information matrix. So $F$ is the Hessian of KL at zero perturbation. KL is asymmetric, but the reverse direction $D_{KL}(\pi_{\theta + p} \,\|\, \pi_\theta)$ has the same Hessian at $p = 0$. The two directions only disagree at third order, so for trust region purposes the choice of which argument is the "old" policy is convention.

In the bandit, all of this is computable by hand. The score $\frac{d}{d\theta} \log \pi_\theta(a)$ is $1 - \sigma(\theta)$ for action 1 and $-\sigma(\theta)$ for action 0, so

$$
F(\theta) = \sigma(\theta)\,(1 - \sigma(\theta))^2 + (1 - \sigma(\theta))\,\sigma(\theta)^2 = \sigma(\theta)(1 - \sigma(\theta)).
$$

One number, largest at $\theta = 0$ and collapsing toward zero at both saturated ends. The local quadratic form of the KL between the policies at $\theta$ and $\theta + p$ is $\tfrac{1}{2} \sigma(1 - \sigma) p^2$: out near saturation, a unit of parameter motion buys almost no policy motion, which is exactly what the sigmoid numbers in the previous section showed. The Fisher is the object that knows this.

The Fisher matrix shows up in policy optimization not because of a coincidence about score functions, but because it is what KL's local quadratic always looks like. If we want a trust region in policy space, we want the constraint to be small KL, and locally that means small $p^\top F p$.

## 7. Natural gradient as a trust region method

Natural Policy Gradient takes the trust region setup and replaces the Euclidean ball with the Fisher ball:

$$
\max_p \, g^\top p \quad \text{s.t.} \quad \tfrac{1}{2} p^\top F p \le \Delta.
$$

KKT gives the closed-form solution:

$$
p^\star = \sqrt{\frac{2 \Delta}{g^\top F^{-1} g}} \, F^{-1} g.
$$

The direction is $F^{-1} g$, which is called the natural gradient. The name is Amari's, from a broader theory of learning on curved parameter spaces. The magnitude is set by the trust radius $\Delta$.

Two ways to think about this. First, it is the trust region step under the right notion of distance. Second, it is a preconditioned gradient where the preconditioner is the inverse Fisher. Directions that produce large policy shifts get damped, directions that barely move the policy get amplified.

Try the second one on the bandit, because the result is worth staring at. The gradient of expected return is

$$
\frac{dJ}{d\theta} = \sigma(\theta)(1 - \sigma(\theta))(r_1 - r_0),
$$

and the Fisher is $\sigma(\theta)(1 - \sigma(\theta))$. Near saturation the two collapse together: at $\theta = 5$ the factor $\sigma(1-\sigma)$ is about $0.0066$, so plain gradient ascent crawls, even though the policy may be nowhere near where we want it. This is not an artifact of the example: the gradient is an average of score vectors weighted by advantages, and the Fisher is the second moment of those same score vectors, so wherever the Fisher collapses, the gradient collapses with it. Dividing by the Fisher removes the dying factor:

$$
F(\theta)^{-1} \frac{dJ}{d\theta} = r_1 - r_0.
$$

A constant. Every trace of $\theta$, of the sigmoid, of the decision to parameterize by logits, is gone, and what remains is the one quantity the problem was ever about: the gap between the two rewards. Same story for the step length. Plugging into the closed form above gives a step of size $\sqrt{2\Delta / \sigma(1-\sigma)}$, which grows near saturation, because out there it takes more parameter motion to spend the same KL budget.

This is what the natural gradient actually buys you: to leading order, the step is invariant to smooth reparameterization. Rescale $\theta$, change basis, swap logits for some other encoding, and the resulting policy moves the same way. The Euclidean methods of the earlier sections cannot make any claim of this kind.

All of this is easier to feel than to read. Drag $\theta$, move the rewards, and watch which readouts care about the parameterization and which don't:

<iframe src="content/bandit-explorer.html" style="width:100%;height:85vh;border:none;border-radius:8px;" loading="lazy"></iframe>

There is a third interpretation that I think is underrated. Under compatible function approximation (where the critic is linear in the score $\nabla_\theta \log \pi_\theta$), the natural gradient direction points toward a greedy policy improvement step rather than merely uphill. This was Kakade's original framing in the 2001 paper. So the natural gradient sits on the policy iteration side of the family, not just the gradient methods side. It is a trust-region method, a preconditioned gradient method, and a soft policy iteration step, all at once.

## 8. TRPO

TRPO is the natural gradient applied to the importance-sampled surrogate from conservative policy iteration (Kakade and Langford, 2002). The setup is: collect data under the current policy $\pi_{\theta_{old}}$, then optimize

$$
L(\theta) = \mathbb{E}_{s, a \sim \pi_{\theta_{old}}}\!\left[ \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{old}}(a \mid s)} \hat A(s, a) \right]
$$

subject to a KL trust region. The full TRPO problem is

$$
\max_\theta \, L(\theta) \quad \text{s.t.} \quad \bar D_{KL}(\theta_{old}, \theta) \le \Delta,
$$

where $\bar D_{KL}$ is the mean KL over states drawn from $\pi_{\theta_{old}}$.

The surrogate is not arbitrary. At $\theta = \theta_{old}$ every importance ratio equals one, and differentiating the ratio pulls out a score: $\nabla_\theta L \big|_{\theta_{old}} = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a \mid s) \, \hat A(s, a)\right]$, the policy gradient itself. So $L$ matches the true return to first order at the expansion point, and you can evaluate it at any candidate $\theta$ using only data already collected under the old policy. It is playing exactly the game $m_k$ played in section 1: a local model, built from information at the current point, trusted only nearby. The KL region says how far.

This is a constrained nonlinear optimization on a neural network policy, which we cannot solve directly. TRPO does the obvious thing: linearize the objective, quadratize the constraint.

1. Linearize $L$ around $\theta_{old}$: $L(\theta_{old} + p) \approx L(\theta_{old}) + g^\top p$, where $g = \nabla_\theta L \big|_{\theta_{old}}$.
2. Quadratize the KL constraint: $\bar D_{KL}(\theta_{old}, \theta_{old} + p) \approx \tfrac{1}{2} p^\top F p$, where $F$ is the Fisher.

This recovers the natural gradient subproblem from the previous section, with closed-form direction $p^\star \propto F^{-1} g$. So TRPO is not a new algorithm grafted onto policy gradients. It is the natural gradient applied to the importance-sampled surrogate.

It is worth noticing where the curvature went. In section 3, the curvature lived in the model $m_k$ and the constraint was a plain ball. Here the objective is linearized, flat, and all of the curvature lives in the constraint. That is a deliberate trade. Second derivatives of the surrogate would have to be estimated from noisy Monte Carlo returns; the curvature of the KL is the Fisher, which is built from first derivatives of $\log \pi_\theta$ alone. TRPO drops the expensive, noisy curvature and keeps the cheap curvature that measures the thing we actually want to control.

The motivation for the trust region beyond "we want the natural gradient direction" is the monotonic improvement bound. Under appropriate assumptions, the true return satisfies

$$
J(\pi_{new}) \ge L_{\pi_{old}}(\pi_{new}) - C \cdot \max_s D_{KL}(\pi_{old}(\cdot \mid s) \,\|\, \pi_{new}(\cdot \mid s))
$$

for a constant $C$ depending on advantage magnitudes and $\gamma$. So if you improve the surrogate while keeping KL small, true return cannot collapse. The actual algorithm uses mean KL instead of max KL, sampled estimates instead of expectations, truncated CG instead of exact $F^{-1}$, GAE instead of true returns, and a neural network policy with no real smoothness guarantees. The bound holds for an idealized procedure, not the code that runs. It supports the design principle (bound policy motion, optimize a surrogate, iterate) without certifying any specific implementation.

## 9. Hessian-vector products and conjugate gradient

The closed-form expression for the natural gradient wants $F^{-1} g$. For a network policy, $F$ has $|\theta| \times |\theta|$ entries, often hundreds of millions on a side. You cannot form $F$, store $F$, or invert $F$. TRPO sidesteps this in two stages.

**Stage one: solve, do not invert.** Conjugate gradient solves $F x = g$ using only the action of $F$ on a vector. Each CG iteration needs one matrix-vector product $F v$, and a few dozen iterations (often around ten in practice) give a high-quality approximate solution. CG is exact in $|\theta|$ steps in exact arithmetic, but in floating point with poorly conditioned $F$ this guarantee does not hold, which is one reason TRPO truncates early regardless.

**Stage two: compute $F v$ without $F$.** This is the Pearlmutter trick (1994), and it is short enough to write out.

We want $F v$ where $F = \nabla_\theta^2 \bar D_{KL}(\theta_{old}, \theta) \big|_{\theta = \theta_{old}}$. Two backward passes do it:

1. Compute $g_{KL}(\theta) = \nabla_\theta \bar D_{KL}(\theta_{old}, \theta)$ as a function of $\theta$ via backprop. (The numerical value at $\theta = \theta_{old}$ is zero, since KL is minimized there. This does not matter, because what we care about is the dependence of $g_{KL}$ on $\theta$, which autograd preserves symbolically.)
2. Form the scalar $s(\theta) = g_{KL}(\theta)^\top v$, where $v$ is treated as a constant numerical vector.
3. Compute $\nabla_\theta s$ via a second backward pass.

Why this gives $F v$: by the product rule,

$$
\nabla_\theta(g_{KL}^\top v) = (\nabla_\theta g_{KL}) v + g_{KL}^\top (\nabla_\theta v).
$$

The first term is the Hessian of $\bar D_{KL}$ acting on $v$, which is exactly $F v$. The second term is zero because $v$ does not depend on $\theta$. So the autograd output is $F v$, in time comparable to one gradient computation.

Combining: CG needs roughly ten iterations, each iteration does one $F v$, each $F v$ costs two backward passes. Total cost is around twenty backward passes for the natural gradient direction. This is what makes TRPO actually run.

## 10. The line search

The closed-form natural gradient step assumes the quadratic approximation of KL is exact. It is not. KL has a third-order term that we threw away, the Fisher is estimated from finite samples, and the surrogate $L$ is itself only a local proxy. If you take the analytic step length, you can land outside the KL ball you intended to respect, and the surrogate may not have actually improved.

So TRPO does a backtracking line search after CG. It starts with the analytic step length, and tries successively shorter step lengths $\alpha \cdot p^\star$ for $\alpha = 1, 1/2, 1/4, \ldots$, accepting the first $\alpha$ for which both

* the actual mean KL is at most $\Delta$, and
* the surrogate $L$ has improved over $\theta_{old}$.

If no $\alpha$ in the schedule works, the update is rejected. This is the same test that $\rho_k$ performed back in section 3, wearing different clothes: compare what the model promised against what actually happened, and refuse to move when the model lied. Without the line search, the trust region is a target rather than a constraint. The line search is what makes it a real constraint, and it is the part of TRPO that gets glossed in most descriptions.

## 11. PPO

PPO inherits TRPO's premise that updates should be local, and throws out the apparatus that enforces it. Instead of a constrained problem with CG and a line search, PPO optimizes the clipped surrogate

$$
L^{CLIP}(\theta) = \mathbb{E}\!\left[ \min\!\left( r_t(\theta) \hat A_t, \, \mathrm{clip}(r_t(\theta), 1 - \varepsilon, 1 + \varepsilon) \hat A_t \right) \right],
$$

where $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{old}}(a_t \mid s_t)$.

The clip is there because PPO squeezes each batch harder than TRPO does. TRPO takes one step per batch of rollouts and then goes back for fresh data. PPO runs several epochs of minibatch steps on the same batch, and over those epochs the ratios drift away from one. Clipping makes the drift self-limiting: once a sample's ratio passes $1 \pm \varepsilon$ in the profitable direction, that sample stops contributing gradient.

What you get: the implementation collapses to minibatch SGD on a single objective. No CG, no Pearlmutter trick, no line search, no Fisher. This is most of why PPO took over.

What you give up: the clipped objective is not a KL constraint. A sample that stops contributing gradient is not a sample whose ratio stops moving: updates driven by the rest of the minibatch can drag it far outside the range, and in practice ratios do end up there. People running PPO monitor the mean KL anyway, and sometimes cut an epoch short when it grows too large. If the clip really did the job of a trust region, nobody would have to watch the KL. The clip is a useful proxy. It is still a proxy.

The PPO paper actually proposed a second variant that keeps KL explicit, as a penalty added to the objective with an adaptive coefficient, a soft version of the TRPO constraint. It lost to the clip on the paper's own benchmarks, and the field followed. Even the variant that kept more of the trust region machinery was outcompeted by the crudest approximation of it.

In the trust region framework, PPO does not have a clean derivation. I know of no choice of distance function and local model that produces the clipped objective by linearization. It is a heuristic that approximates the behavior of a trust region method without inheriting its structure. The honest statement is that PPO kept the trust region philosophy and dropped the trust region mechanism.

## 12. Summary

The story compresses to:

1. Iterative methods for nonlinear optimization need a way to control how far they trust the local quadratic model. Trust regions enforce this by constraining the step to a region where the model is believed.
2. The Euclidean trust region in parameter space is wrong for policies because Euclidean distance in $\theta$ does not measure how much the distribution $\pi_\theta$ changed.
3. KL divergence does measure that. Locally, KL is quadratic in the parameter perturbation with Hessian equal to the Fisher information matrix.
4. The natural gradient is steepest ascent under the Fisher metric. Closed-form direction $F^{-1} g$. In the two-action bandit it comes out to the constant $r_1 - r_0$: the parameterization deleted from the update, leaving only the quantity the problem is about.
5. TRPO is the natural gradient applied to the importance-sampled surrogate, computed using conjugate gradient on Hessian-vector products, with a backtracking line search to actually enforce the KL constraint.
6. PPO drops the constraint and the line search and replaces them with a clipped surrogate. Easier to run, weaker as a trust region method.

The reason the trust region perspective is worth holding onto, even if you only ever run PPO, is that it tells you what is being controlled. The dangerous quantity in policy optimization is policy motion, not parameter motion. If a method has a clean answer to the question "what is the local model and what notion of distance is being bounded", it is a trust region method. Otherwise, it is an approximation to one. The methods that train today's language models are descendants of PPO, and the same two questions cut through each of them: what distance is being bounded, and what actually enforces the bound. That is [the next post](/blog/two-kls/).

## References

1. Jorge Nocedal and Stephen Wright. Numerical Optimization. Springer, 2nd edition, 2006.
2. Shun-ichi Amari. Natural Gradient Works Efficiently in Learning. Neural Computation, 1998.
3. Sham Kakade. A Natural Policy Gradient. NIPS, 2001.
4. Sham Kakade and John Langford. Approximately Optimal Approximate Reinforcement Learning. ICML, 2002.
5. John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, Philipp Moritz. Trust Region Policy Optimization. ICML, 2015.
6. Barak Pearlmutter. Fast Exact Multiplication by the Hessian. Neural Computation, 1994.
7. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.

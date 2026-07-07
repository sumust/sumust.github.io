# The Trust Region Subproblem

*Part 2 of a five-part series on trust regions: [1. Rosenbrock Explorer](/blog/rosenbrock-explorer/) · [2. The Trust Region Subproblem](/blog/trust-region-subproblem/) · [3. Trust Regions, Natural Gradients, TRPO, PPO](/blog/trust_regions/) · [4. The Two KLs](/blog/two-kls/) · [5. GRPO, or What the Critic Was For](/blog/grpo/)*

The [next post in this series](/blog/trust_regions/) needs exactly one fact about the trust region subproblem: that you can solve it approximately with conjugate gradient and stop early. This post is the subproblem done properly: what the exact solution looks like, the two classical approximations, and why the crude one secretly carries the convergence theory.

## 1. The problem

At the current iterate you have a quadratic model of the objective,

$$
m(p) = J + g^\top p + \tfrac{1}{2} p^\top B p,
$$

built from the value, the gradient, and the Hessian or an approximation to it. The model is trustworthy only near the current point, so instead of minimizing it outright you minimize it inside a ball:

$$
\min_p \, m(p) \quad \text{s.t.} \quad \|p\| \le \Delta.
$$

The radius $\Delta$ is managed from outside: grow it when the model's predictions keep coming true, shrink it when they don't. That loop is the trust region *method*, and it is covered in the next post. Here the question is just: given $g$, $B$, and $\Delta$, how do you actually compute the step?

Assume throughout that $B$ is positive definite. Then there are two regimes. If the unconstrained minimizer of the model, the Newton step $p^N = -B^{-1} g$, satisfies $\|p^N\| \le \Delta$, take it; the constraint is slack and there is nothing more to do. Otherwise the solution lies on the boundary $\|p\| = \Delta$, and everything below is about that case.

## 2. What the exact solution looks like

The boundary case has a clean characterization: $p^\star$ solves the subproblem exactly when there is a scalar $\lambda \ge 0$ with

$$
(B + \lambda I)\, p^\star = -g, \qquad \|p^\star\| = \Delta.
$$

The multiplier $\lambda$ is the price of the constraint. At $\lambda = 0$ you recover the Newton step. As $\lambda \to \infty$ the identity term dominates and $p^\star \approx -g / \lambda$: a short step in the steepest descent direction. In between, the family $p(\lambda) = -(B + \lambda I)^{-1} g$ traces a curve from the Newton step to the steepest descent direction, and the exact trust region solution is the point on that curve at distance $\Delta$. This is the same family of steps that Levenberg-Marquardt uses for least squares, which is not a coincidence: damping the Hessian and constraining the step are two views of the same operation.

Solving exactly means finding the $\lambda$ where $\|p(\lambda)\| = \Delta$, a one-dimensional root-finding problem. The catch is the cost per evaluation: each candidate $\lambda$ requires factoring $B + \lambda I$. The standard safeguarded Newton iteration on $\lambda$ (due to Moré and Sorensen) typically needs two or three factorizations, which is fine when you can factor at all. For a neural network you cannot, and that is what sections 4 and 5 are for.

## 3. The Cauchy point

The cheapest serious answer: minimize the model along the steepest descent ray, and truncate at the boundary if the minimizer falls outside. Closed form:

$$
p^C = -\tau \frac{\Delta}{\|g\|} g, \qquad
\tau = \begin{cases}
1 & \text{if } g^\top B g \le 0, \\
\min\left(1, \frac{\|g\|^3}{\Delta \, g^\top B g}\right) & \text{otherwise.}
\end{cases}
$$

The Cauchy point ignores curvature in every direction except the one it moves along, and as a step it is roughly as slow as steepest descent. So it looks like a throwaway. It is not, and this is the part I find satisfying: the global convergence theory of trust region methods rests on the Cauchy point, not on the clever steps. The standard theorems require only that each accepted step achieves some fixed fraction of the model decrease the Cauchy point achieves. Any method that clears that bar (dogleg, truncated CG, exact solves) inherits global convergence. The crude step carries the theory. The clever steps carry the speed.

## 4. Dogleg

Powell's dogleg method builds a two-segment path and walks it until it hits the boundary. The first segment runs from the zero step to $p^U$, the unconstrained minimizer of the model along the steepest descent direction. The second runs from $p^U$ to the Newton step $p^N$. When $\Delta$ is small the step is mostly along the gradient leg; when $\Delta$ is large it gets close to the Newton point.

The reason this particular path: as $\Delta$ grows from zero to infinity, the *exact* solution of the subproblem traces a curve that leaves the origin along $-g$ and bends toward the Newton step. It is the curve from section 2, reparameterized by radius instead of by $\lambda$. The dogleg is that curve approximated by two line segments.

Finding where the path crosses the boundary is a scalar quadratic equation. On the second leg, solve for $t \in [0, 1]$ in

$$
\|t\, p^N + (1 - t)\, p^U\|^2 = \Delta^2.
$$

For this to make sense, the path needs to cross the boundary exactly once, and that is guaranteed by a fact worth proving: if $B$ is positive definite, then $\|p^U\| \le \|p^N\|$, and more strongly, the distance from the origin increases monotonically along the whole dogleg path. The norm comparison is a Cauchy-Schwarz consequence. Write everything in the eigenbasis of $B$ and normalize the gradient coordinates into probability weights; the inequality reduces to

$$
1 \le \left(\mathbb{E}[\mu]\right)^2 \mathbb{E}\!\left[1/\mu^2\right],
$$

where $\mu$ is an eigenvalue drawn with those weights, and that holds by applying Cauchy-Schwarz twice. So when both candidate points are outside the region, the unconstrained steepest-descent minimizer is always the closer one, the path never doubles back, and the boundary crossing is unique. Dogleg requires $B \succ 0$; without it $p^N$ is not a descent target and the second leg makes no sense.

Here is the whole section in one picture. The green curve is the family of exact solutions as the radius sweeps from zero to infinity, the two-segment path is the dogleg, and the circle is the trust region. Drag the radius and watch what the two segments give up:

<iframe src="content/subproblem-explorer.html" style="width:100%;height:85vh;border:none;border-radius:8px;" loading="lazy"></iframe>

## 5. Truncated conjugate gradient

At neural-network scale you cannot form $B$, factor $B$, or store $B$. What you can do is compute $Bv$ for a given vector $v$; for a Hessian this costs about one extra backward pass ([the next post](/blog/trust_regions/) spells out how). That is exactly the interface conjugate gradient wants.

The method of Steihaug and Toint: run CG on the linear system $Bp = -g$, starting from the zero step, and stop early if either of two things happens: the iterate leaves the trust region, or CG discovers a direction of negative curvature. In both cases, follow the current search direction to the boundary and return that.

Two properties make this more principled than "run CG and hope":

* The first CG iterate is exactly the Cauchy point. So even one iteration clears the bar that the convergence theory sets, and every further iteration only decreases the model.
* Started from zero, the CG iterates move monotonically away from the origin. Like the dogleg path, the trajectory crosses the boundary at most once, so stopping at the crossing is well defined.

The result lands somewhere between the Cauchy point and the Newton step, depending on how many iterations you allow, which makes truncated CG a kind of adaptive dogleg: same endpoints, but the path bends using real curvature information instead of two straight legs.

## 6. Which one to use

Exact solves when the problem is small enough to factor and you need the accuracy. Dogleg when $B$ is positive definite by construction (quasi-Newton approximations, Gauss-Newton) and factoring once per step is affordable. Truncated CG when the only thing you can afford is matrix-vector products, which is the neural network regime, and which is why it is the version that appears inside TRPO [in the next post](/blog/trust_regions/).

## References

1. Jorge Nocedal and Stephen Wright. Numerical Optimization. Springer, 2nd edition, 2006. Chapter 4 covers everything here, including the Moré-Sorensen iteration, the dogleg lemma, and the Steihaug-Toint method.

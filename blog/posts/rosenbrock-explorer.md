# Rosenbrock Explorer

*Part 1 of a five-part series on trust regions: [1. Rosenbrock Explorer](/blog/rosenbrock-explorer/) · [2. The Trust Region Subproblem](/blog/trust-region-subproblem/) · [3. Trust Regions, Natural Gradients, TRPO, PPO](/blog/trust_regions/) · [4. The Two KLs](/blog/two-kls/) · [5. GRPO, or What the Critic Was For](/blog/grpo/)*

The [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function) is one of the most well-known test problems in optimization:

$$
f(x, y) = (1 - x)^2 + 100 \, (y - x^2)^2,
$$

minimized at $(1, 1)$. The 100 is the whole personality of this function. The second term punishes leaving the parabola $y = x^2$ a hundred times harder than the first term punishes being far from the solution along it. The result is a long, narrow, curved valley: easy to fall into, hard to travel along.

That shape is precisely what breaks first-order methods. Almost everywhere, the gradient points across the valley, not along it, so steepest descent spends its effort bouncing between the walls while creeping toward the minimum. The severity of this is measured by the condition number of the Hessian, the ratio of the largest to smallest eigenvalue of the local curvature. At the minimum itself it is about 2500: the bowl is 2500 times stiffer in one direction than the other. Newton's method rescales the gradient by the inverse of that curvature, which is why its arrow points calmly along the valley while the gradient arrow slams into the wall.

The explorer below lets you drag a point around the function's contour plot and watch several quantities update in real time:

- **Gradient** — direction and magnitude of steepest ascent.
- **Hessian eigenvalues** — how stretched the local curvature is (the condition number κ tells you how much steepest descent will zigzag).
- **Newton step vs. steepest descent** — two arrows show where each method would go next. Notice how Newton aims for the valley floor while steepest descent overshoots across it.

Some things worth trying:

- Park the point at $(-1.2, 1)$, the classical benchmark start. The two arrows disagree almost completely about what to do.
- Slide along the valley floor $y = x^2$ and then nudge just off it. The gradient magnitude jumps by orders of magnitude for a visually tiny move.
- Watch the $f(x+p)$ readout, which is the function value where the Newton step lands. From most of the valley, a single Newton step gets you close to the floor.

<iframe src="content/rosenbrock-explorer.html" style="width:100%;height:85vh;border:none;border-radius:8px;" loading="lazy"></iframe>

Newton looks unbeatable here, and on this function it mostly is. But the full Newton step is an act of trust in a quadratic model that is only honest near the current point, and Rosenbrock is gentle enough to forgive that trust. The rest of this series is about what to do when the model does not deserve it: [part 2](/blog/trust-region-subproblem/) works out how to take the best step inside a region where the model can be believed, and [part 3](/blog/trust_regions/) follows that idea all the way to TRPO and PPO.

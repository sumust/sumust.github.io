# Rosenbrock Explorer

The [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function) is one of the most well-known test problems in optimization. Its narrow, curved valley makes it a great example for visualizing why some optimization methods struggle and others don't.

The **[Rosenbrock Explorer](content/rosenbrock-explorer.html)** is a small interactive tool that lets you drag a point around the function's contour plot and watch several quantities update in real time:

- **Gradient** — direction and magnitude of steepest ascent.
- **Hessian eigenvalues** — how stretched the local curvature is (the condition number κ tells you how much steepest descent will zigzag).
- **Newton step vs. steepest descent** — two arrows show where each method would go next. Notice how Newton aims for the valley floor while steepest descent overshoots across it.

It's a useful way to build intuition for why second-order information matters, especially in ill-conditioned landscapes. Give it a try — just click and drag.

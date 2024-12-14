Documentation of code for Optimized Belief Propagation (CPU and GPU)

Project page with code and README: https://github.com/sgrauerg6/cudaBeliefProp

Code is structured such as evaluation portion is separate from implementation
and also using parent/child classes in the belief propagation implementation so
the CPU/GPU implementations share the same code where possible; the framework is
there to add optimized belief propagation implementations for additional
architectures and also to use the evaluation code for evaluation of other
benchmarks.

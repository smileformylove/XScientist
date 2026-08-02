# Performance Regression Gates

Repository simplification must not make XScientist slower, heavier at runtime,
or less scientifically capable. Functional tests remain the primary behavior
gate; cold-import and memory measurements provide an additional engineering
gate for changes to packaging, dependencies, and orchestration.

Record a baseline before a simplification round:

```bash
python tools/performance_regression.py record \
  --output /tmp/xscientist-before.json \
  --repeats 9
```

Record the candidate under the same interpreter, machine, and checkout posture,
then compare:

```bash
python tools/performance_regression.py record \
  --output /tmp/xscientist-after.json \
  --repeats 9

python tools/performance_regression.py compare \
  --baseline /tmp/xscientist-before.json \
  --candidate /tmp/xscientist-after.json
```

The default gate permits at most 5% median cold-import and peak-RSS regression,
plus a small absolute noise allowance. Baseline and candidate environments must
match exactly. For production-path refactors, also require:

- identical deterministic artifact and gate decisions for fixed inputs;
- no increase in LLM calls, token budgets, retry limits, or fallback rate;
- unchanged experiment concurrency and timeout semantics;
- the full unit, protocol, packaging, and repository validation suites;
- real-project canaries before removing a compatibility path.

The public CLI has a structural test that forbids eager loading of heavy model,
data, plotting, and provider SDK dependencies. This guards startup performance
more reliably than a wall-clock threshold alone.

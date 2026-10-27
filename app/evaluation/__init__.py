"""Routing evaluation (paper Section 6.4): the online protocol, metrics and the benchmarks built on them.

Nothing here runs inside the service. The benchmarks are reported separately and never merged:

* ``legacy``    -- the original 123-scenario, 14-intent benchmark (``scripts/eval_routing.py``);
* ``corrected`` -- the same benchmark with the weak-match semantics enforced (``scripts/eval_routing.py``);
* ``paper968``  -- the 96-scenario, Table-2 benchmark built from Splunk Attack Data metadata
  (:mod:`app.evaluation.paper968`, ``scripts/eval_routing_paper.py``).
"""

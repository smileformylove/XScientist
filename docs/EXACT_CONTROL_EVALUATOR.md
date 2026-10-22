# Exact Supplied-Model Control Evaluation

`ai_scientist.utils.exact_control_evaluator` evaluates a small supplied finite
model and immutable public controller with exact rational arithmetic. It is
separate from the asserted-trace aggregation in `control_evaluator.py`.

## Contract

- `FiniteControlModel` specifies reset probabilities, transitions
  `T[action][state][next_state]`, emissions `E[state][observation]`, bounded
  rewards `reward[action][next_observation]`, and `initial_reward[observation]`.
- `PublicController` stores layers of nodes. Each node contains action
  probabilities and successor indices indexed by action and next observation.
  Initial node indices depend only on the initial observation. All pointers,
  including impossible and zero-policy-probability branches, are validated.
- There are K actions and K+1 observations. Last-layer successors are -1.
  K=0 includes only the initial observation reward. Protocol v2 maps to K=H,
  zero initial reward. The short-reset theorem maps to K=H-1, both initial
  reward and next-observation reward `(1+X)/2`. These are different interfaces.
- Reward is determined by public action/observation, so it adds no information.
  Additional informative, hidden-state, stochastic or terminal rewards are
  not supported. Do not silently discard them when adapting an environment.
- Exact `int` or `Fraction` inputs only; floats and booleans are rejected.
  Probability rows sum exactly to one. Rewards lie in [-1,1]. Model dimensions,
  input rational bits, controller nodes, horizon and tree work are bounded.
  Distinct input denominators have a shared 4096-bit budget; intermediate
  numerators and denominators are checked against an 8192-bit limit during
  accumulation, before report generation. Repeated denominators count once.
  This is a diagnostic evaluator, not a scalable general POMDP solver.

## Calculation

An exact belief Bellman recursion maximizes over public-history policies, not
hidden-state policies. A separate forward hidden-state/node mass calculation
replays the supplied controller, including its stored successor pointers.
It never re-filters using true parameters to replace the controller's state.

On learner histories, `optimal_action_loss = V*(history)-Q*(history,action)`.
The evaluator integrates this loss with learner occupancy and checks exactly:

```text
sum_action expected_count = K
sum_action regret_contribution = V* - V_controller
```

Unused actions have a null conditional loss, not an observed zero. The vector
is diagnostic; its sum is ordinary regret and is not a second efficacy endpoint.

## Evidence Boundary

Reports bind canonical model, policy, implementation and report hashes.
Computed optimality and return are conditional on the supplied mathematical
model. They do not verify the model's origin, a real environment, a dataset,
an empirical learning rate, or policy performance outside that model.

`comparison_ready`, experiment/training authorization and claim promotion
remain false. This report is not connected to generic evidence-admission
bindings. An independent trusted environment, admitted protocol and fresh
training remain separate prerequisites for an empirical scientific claim.

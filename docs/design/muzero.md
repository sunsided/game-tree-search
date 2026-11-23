# Search over a learned model (MuZero)

Design note for [#52](https://github.com/sunsided/game-tree-search/issues/52): what it would
take for the crate's Monte Carlo tree search to plan with a learned model, as MuZero and
Stochastic MuZero do, instead of the game's own rules. Nothing here is built yet; this note
fixes the shape so that the first user with a network can start from it.

The reference is DeepMind's mctx (Apache-2.0, © 2021 DeepMind Technologies Limited), whose
`muzero_policy`, `gumbel_muzero_policy` and `stochastic_muzero_policy` implement the three
algorithms. Every interface and formula quoted from mctx below comes from its `main` branch:
`base.py`, `search.py`, `tree.py`, `qtransforms.py` and `policies.py`.

## What changes compared with today

Today's [`Mcts`](../../src/mcts.rs) plans with a [`Rewarded`](../../src/mcts.rs) game:

- it reaches a node by replaying the moves from the root with `Game::play`;
- it lists legal moves with `Game::moves`;
- a reward exists only at a leaf, `[0, 1]` from Max's view;
- every node's value is the mean of those rewards.

A learned model breaks each of these assumptions.

| Today (`Game`) | Learned model |
|---|---|
| positions replayed from the root by `play` | hidden states the model produced, which cannot be rebuilt without calling the model again |
| legal moves per position | a fixed action space; legality known only at the root |
| reward only at the end, in `[0, 1]` | a predicted reward and discount on every transition, of any scale |
| side to move from the position | whose turn it is must come from the observation or the model |
| leaf value from a rollout or the evaluation | value and priors from the prediction function, never a rollout |

## The model interface in mctx

mctx calls the model through two functions.

- **Representation**, at the root, returns a `RootFnOutput` with `prior_logits`, `value`
  ("an approximate value of the current state") and `embedding` ("the inputs to the next
  `recurrent_fn` call").
- **Dynamics plus prediction**, the `RecurrentFn`, takes `(params, rng_key, action,
  embedding)` and returns `(RecurrentFnOutput, new_embedding)`. `RecurrentFnOutput` holds:
  - `reward`: "an approximate reward from the state-action transition";
  - `discount`: "the discount between the `reward` and the `value`";
  - `prior_logits`;
  - `value`: "an approximate value of the state after the state-action transition".

The search stores the reward and the discount on the edge (`expand` in `search.py`), and
the embedding with the new node. It computes a Q-value from them (`_unbatched_qvalues` in
`tree.py`):

```text
Q(parent, a) = children_rewards[parent, a] + children_discounts[parent, a] * children_values[parent, a]
```

It backs a leaf's value up one edge at a time (`backward` in `search.py`):

```text
leaf_value   = reward + discount * leaf_value
parent_value = (parent_value * count + leaf_value) / (count + 1)
```

Since rewards and values have no fixed range, selection does not read Q directly. mctx's
MuZero policy normalises it per node with `qtransform_by_parent_and_siblings`: the min and
max are taken over the node's own value and its children's Q-values, and unvisited children
get the minimum, which maps to 0. The policy's PUCT constants default to
`pb_c_init = 1.25` and `pb_c_base = 19652`, and Dirichlet noise to `dirichlet_alpha = 0.3`
and `dirichlet_fraction = 0.25`. Gumbel MuZero uses `qtransform_completed_by_mix_value`
instead, the transform `search_gumbel` already implements.

Stochastic MuZero splits each transition in two:

- **Decision step.** `DecisionRecurrentFnOutput` maps a state and an action to an
  afterstate, with "no discount factor or reward", and gives `chance_logits` over `C`
  chance outcomes and an `afterstate_value`.
- **Chance step.** `ChanceRecurrentFnOutput` maps an afterstate and a chance outcome to the
  next state, with `action_logits`, `value`, `reward` and `discount`.

## Proposed shape

### A model trait beside `Game`

```rust
/// A learned model to plan with: a representation for the root, and dynamics with
/// prediction for every step after it.
pub trait Model {
    /// What the model encodes the root from.
    type Observation;
    /// The model's hidden state of a position.
    type State;

    /// The number of actions; actions are `0..actions()`.
    fn actions(&self) -> usize;

    /// The root's hidden state, with its prediction.
    fn represent(&mut self, observation: &Self::Observation) -> (Self::State, Prediction);

    /// The state `action` leads to from `state`, with the step's reward and discount and the
    /// new state's prediction.
    fn dynamics(&mut self, state: &Self::State, action: usize) -> (Self::State, Step, Prediction);
}

/// What the prediction function says about a state: a prior per action, its value from
/// Max's view, and whose turn it is.
pub struct Prediction { pub priors: Vec<f64>, pub value: f64, pub to_play: Side }

/// What a transition earned: its reward, and the discount applied to what follows.
pub struct Step { pub reward: f64, pub discount: f64 }
```

- It is `&mut self`, like `Evaluator`, so a model can keep scratch buffers or a device
  handle. A batched variant mirrors `BatchEvaluator`: many `(state, action)` pairs in, as
  many `(state, step, prediction)` triples out. That keeps the search from assuming one
  synchronous model call per leaf.
- Actions are a fixed index range, as in mctx. Legal actions are a mask at the root only,
  given with the observation. The model is trusted below the root.
- Values and rewards are on the model's scale and from Max's view. The mctx code read for
  this note has no notion of sides; it backs up `reward + discount * value` and nothing
  else. `Prediction::to_play` says whose turn it is, and selection compares from that side,
  as `NodeStats::mean_for` does today.

### A separate search on the same plug points

A `ModelMcts` next to `Mcts`, not a mode of it, because node storage differs: each node
owns its hidden state, and each edge its reward and discount, instead of being replayed
from the root. It reuses:

- `Selection` with `Puct`, and `search_gumbel`'s halving, fed normalised Q-values. The
  min-max normalisation of mctx's `qtransform_by_parent_and_siblings` maps them into the
  `[0, 1]` that `NodeStats::mean_for` and the selection rules already expect.
- `RootNoise`, `MctsResult` (`policy`, `improved`, `sample`) and the `Random` sources.
- The batched path's pending marks and virtual loss, for a batched model.

`Mcts` itself stays as it is: games with real rules keep exact transitions and rollouts.

### Testing without a network

- An exact model of a real game: the state is the game's position, `dynamics` plays the
  move, the reward is the end-of-game reward and the discount is 1. `ModelMcts` must match
  `Mcts` with `StaticValue` on that game, move for move, given the same priors.
- A one-player chain with rewards along the way and a discount below 1, whose discounted
  returns can be computed by hand, checks the backup formula above.
- A model whose values are scaled by 1000 must choose the same moves, which checks the
  normalisation.

## Stochastic MuZero

The crate already has chance nodes for exact games (`Stochastic`, `expectimax`).
Stochastic MuZero would add a second, afterstate level to `ModelMcts`: decision nodes
select actions as above, and chance nodes choose an outcome by its `chance_logits`. This
note does not settle how mctx's chance nodes choose an outcome, since that code was not
read for it. Building this would start there.

## Out of scope

- Training, networks, replay buffers and observation encoders. They belong in a companion
  crate, as #5 planned, with an optional Burn or Candle backend.
- Imperfect-information games, as before.

## When to build it

When someone brings a model to plan with: a trained network, or at least a toy one with a
real training loop around it. Without one, the interface would be guessed. The first
step would then be the `Model` trait, `ModelMcts` with PUCT and the min-max normalisation,
and the exact-model test above. Gumbel and batching would come next, and Stochastic MuZero
last.

## Decisions

The three questions this note left open are settled as follows.

1. **Action indices, `0..actions()`, not the game's move type.** A policy network outputs a
   fixed-size vector, so priors are indexed by action anyway, and legality is known only at
   the root. The adapter for a real game, which the exact-model test needs, gets an
   `action_of(&G::Move) -> usize` mapping to translate moves and back.
2. **`to_play: Side` in `Prediction`, with every value and reward from Max's view.** This is
   the crate's convention (`Rewarded`, `NodeStats::mean_for`), so `Puct` and the Gumbel
   halving compare from the right side unchanged, and games where one side moves several
   times in a row work. A one-player model answers `Side::Max` everywhere.
3. **Every node stores its hidden state, as mctx does.** Re-running the dynamics from the
   root would cost about depth times simulations model calls per search. At the 800 to
   2,000 simulations per move of AlphaZero-style setups, hidden states of a few KB stay in
   the single-digit megabytes. If memory ever matters, the cap is on the tree's size, not
   on recomputation.

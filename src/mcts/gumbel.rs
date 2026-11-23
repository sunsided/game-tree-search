//! Gumbel root selection (Danihelka et al., "Policy improvement by planning with Gumbel",
//! ICLR 2022), ported from the Gumbel `MuZero` policy of `DeepMind`'s mctx: sequential halving
//! with Gumbel noise at the root, and a deterministic selection towards the improved policy
//! below it. It improves the policy even with very few simulations.

use alloc::vec;
use alloc::vec::Vec;

use super::policy::Evaluator;
use super::random::Random;
use super::selfplay::unit;
use super::tree::Tree;
use super::{Mcts, MctsResult, Rewarded};
use crate::Side;

/// The settings of [`Mcts::search_gumbel`]; the defaults are those of mctx.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gumbel {
    /// The most root moves sequential halving considers, `16` by default; fewer if the root
    /// has fewer moves. The search spends its simulations on these, halving them round by
    /// round.
    pub considered: usize,
    /// The scale of the Gumbel noise on the root moves, finite and non-negative; `1` by
    /// default. `0` makes the search deterministic, as for evaluating a trained policy in a
    /// perfect-information game.
    pub scale: f64,
    /// Added to the most visits among a node's children when scaling its Q-values, finite
    /// and non-negative; `50` by default (`c_visit` in the paper, `maxvisit_init` in mctx).
    pub visit_offset: f64,
    /// The scale of the Q-values against the log priors, finite and non-negative; `0.1` by
    /// default (`c_scale` in the paper, `value_scale` in mctx).
    pub value_scale: f64,
}

impl Default for Gumbel {
    fn default() -> Self {
        Self {
            considered: 16,
            scale: 1.0,
            visit_offset: 50.0,
            value_scale: 0.1,
        }
    }
}

impl Gumbel {
    /// Whether every scale and offset is finite and non-negative.
    fn is_valid(&self) -> bool {
        [self.scale, self.visit_offset, self.value_scale]
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    }
}

impl<G: Rewarded, E: Evaluator<G>, S: super::Selection, R: Random> Mcts<G, E, S, R> {
    /// Searches `position` with exactly `simulations` simulations, choosing the root move by
    /// sequential halving with Gumbel noise (Gumbel `MuZero`), and returns that move with the
    /// improved policy as the training target.
    ///
    /// The root's moves get Gumbel noise `g` from this searcher's generator. The search
    /// considers the [`Gumbel::considered`] moves with the highest `g + log prior` and spends
    /// the simulations on them in rounds, keeping the better half by
    /// `g + log prior + σ(Q)` after each round. Below the root, each simulation takes the
    /// child whose visit share lags most behind the improved policy
    /// `softmax(log prior + σ(Q))`, without an exploration constant. A move never visited
    /// counts with a mix of the node's own evaluation and its visited moves' values. `σ`
    /// rescales the Q-values to `[0, 1]` and multiplies them by
    /// `(visit_offset + most visits) * value_scale`.
    ///
    /// The result's [`best`](MctsResult::best) is the move that survived the halving, which
    /// need not be the earliest of the most visited; the line starts with it.
    /// [`MctsResult::improved`] holds the improved policy, which [`MctsResult::policy`]
    /// returns; the visit counts follow the halving schedule and are no training target.
    /// Priors come from the searcher's [`Evaluator`], and the root's own evaluation, not a
    /// simulation, is asked once before the first simulation.
    ///
    /// The search always starts a new tree: it ignores a tree kept by [`Mcts::advance`], and
    /// the [root noise](Mcts::with_root_noise), which the Gumbel noise replaces.
    ///
    /// # Panics
    ///
    /// Panics if a scale or offset of `gumbel` is negative or not finite.
    ///
    /// # Examples
    ///
    /// ```
    /// # use game_tree_search::{Game, Side};
    /// # struct Nim;
    /// # impl Game for Nim {
    /// #     type Position = (u32, Side);
    /// #     type Move = u32;
    /// #     type Score = i32;
    /// #     fn side_to_move(&self, &(_, side): &Self::Position) -> Side { side }
    /// #     fn moves(&self, &(stones, _): &Self::Position, moves: &mut Vec<u32>) {
    /// #         moves.extend((1..=2).filter(|&take| take <= stones));
    /// #     }
    /// #     fn play(&self, &(stones, side): &Self::Position, take: &u32) -> Self::Position {
    /// #         (stones - take, if side == Side::Max { Side::Min } else { Side::Max })
    /// #     }
    /// #     fn evaluate(&self, &(stones, side): &Self::Position) -> i32 {
    /// #         match (stones, side) { (0, Side::Max) => -1, (0, Side::Min) => 1, _ => 0 }
    /// #     }
    /// # }
    /// # impl game_tree_search::Rewarded for Nim {
    /// #     fn reward(&self, score: i32) -> f64 { f64::from(score + 1) / 2.0 }
    /// # }
    /// use game_tree_search::{Gumbel, Mcts};
    ///
    /// let result = Mcts::new().search_gumbel(&Nim, &(7, Side::Max), 200, &Gumbel::default());
    /// assert_eq!(result.best, Some(1));
    /// let improved = result.improved.expect("a Gumbel search sets it");
    /// assert!(improved[0] > improved[1]);
    /// ```
    pub fn search_gumbel(
        &mut self,
        game: &G,
        position: &G::Position,
        simulations: u32,
        gumbel: &Gumbel,
    ) -> MctsResult<G::Move> {
        assert!(
            gumbel.is_valid(),
            "Gumbel scales and offsets must be finite and non-negative: {gumbel:?}"
        );

        self.tree.reset(game, position, &mut self.evaluator);
        self.tree.kept = false;
        let moves = self.tree.children(0).len();
        if moves == 0 {
            return self.tree.result();
        }

        let raw = self.evaluator.evaluate(game, position, &mut self.rng).value;
        self.tree.set_raw(0, raw);
        let root = Root::new(&self.tree, gumbel.scale, &mut self.rng);
        let schedule = considered_visits(gumbel.considered.min(moves), simulations);
        let mut q = Vec::new();
        for &considered in &schedule {
            let leaf = self
                .tree
                .select_by(game, position, &mut self.evaluator, |tree, index| {
                    if index == 0 {
                        root.choose(tree, Some(considered), gumbel, &mut q)
                    } else {
                        choose_interior(tree, index, gumbel, &mut q)
                    }
                });
            let value = self.leaf_value(game, position, &leaf);
            self.tree.backup(value);
        }

        let best = root.choose(&self.tree, None, gumbel, &mut q);
        let mut result = self.tree.result_from(best);
        result.improved = Some(root.improved(&self.tree, gumbel, &mut q));

        result
    }
}

/// The root's Gumbel noise and normalised log priors, drawn once per search.
struct Root {
    /// The noise of each root move, in move order.
    noise: Vec<f64>,
    /// Each root move's log prior minus the largest one.
    logits: Vec<f64>,
}

impl Root {
    /// Draws noise scaled by `scale` for the root moves of `tree` from `rng`.
    fn new<M: Clone>(tree: &Tree<M>, scale: f64, rng: &mut impl Random) -> Self {
        let children = tree.children(0);
        let noise = children.clone().map(|_| scale * gumbel(rng)).collect();
        let mut logits: Vec<f64> = children.map(|child| log_prior(tree, child)).collect();
        let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for logit in &mut logits {
            *logit -= top;
        }

        Self { noise, logits }
    }

    /// The root move to visit among those with `considered` visits, or with the most visits
    /// for `None`: the highest `g + log prior + σ(Q)`, ties going to the earliest.
    fn choose<M: Clone>(
        &self,
        tree: &Tree<M>,
        considered: Option<u32>,
        gumbel: &Gumbel,
        q: &mut Vec<f64>,
    ) -> Option<usize> {
        completed_q(tree, 0, gumbel, q);
        let children = tree.children(0);
        let visits = |child: usize| tree.stats(child).visits;
        let wanted = considered.unwrap_or_else(|| children.clone().map(visits).max().unwrap_or(0));
        let score =
            |offset: usize| (self.noise[offset] + self.logits[offset] + q[offset]).max(-1e9);
        let first = children.start;

        // The schedule always leaves a move with the wanted visits; all moves otherwise.
        let any = children.clone().any(|child| visits(child) == wanted);
        let eligible = children.filter(|&child| !any || visits(child) == wanted);

        argmax(eligible, |child| score(child - first))
    }

    /// The improved policy over the root moves: `softmax(log prior + σ(Q))`.
    fn improved<M: Clone>(&self, tree: &Tree<M>, gumbel: &Gumbel, q: &mut Vec<f64>) -> Vec<f64> {
        completed_q(tree, 0, gumbel, q);
        let mut policy: Vec<f64> = self
            .logits
            .iter()
            .zip(q.iter())
            .map(|(l, q)| l + q)
            .collect();
        softmax(&mut policy);

        policy
    }
}

/// The child of `index`, below the root, whose visit share lags most behind the improved
/// policy: the highest `π'(a) - N(a) / (1 + Σ N)`, ties going to the earliest; `None` where
/// the game ends.
fn choose_interior<M: Clone>(
    tree: &Tree<M>,
    index: usize,
    gumbel: &Gumbel,
    q: &mut Vec<f64>,
) -> Option<usize> {
    tree.side(index)?;
    completed_q(tree, index, gumbel, q);
    let children = tree.children(index);
    let first = children.start;
    for (offset, child) in children.clone().enumerate() {
        q[offset] += log_prior(tree, child);
    }

    softmax(q);
    let total: u32 = children.clone().map(|child| tree.stats(child).visits).sum();
    let share = |child: usize| f64::from(tree.stats(child).visits) / (1.0 + f64::from(total));

    argmax(children, |child| q[child - first] - share(child))
}

/// Writes `σ(completed Q)` of `index`'s children, for the side to move at `index`, into
/// `out`: a visited child's mean reward, an unvisited one's the mixed value of `index`,
/// rescaled to `[0, 1]` over the children, then multiplied by
/// `(visit_offset + most visits) * value_scale`.
fn completed_q<M: Clone>(tree: &Tree<M>, index: usize, gumbel: &Gumbel, out: &mut Vec<f64>) {
    let side = tree.side(index).unwrap_or(Side::Max);
    let children = tree.children(index);
    let (mut visits, mut most, mut mass, mut weighted) = (0u32, 0u32, 0.0, 0.0);
    for child in children.clone() {
        let stats = tree.stats(child);
        visits = visits.saturating_add(stats.visits);
        most = most.max(stats.visits);
        if stats.visits > 0 {
            let prior = stats.prior.max(f64::MIN_POSITIVE);
            mass += prior;
            weighted += prior * stats.mean_for(side);
        }
    }

    let own = tree.raw(index).map_or(0.5, |raw| for_side(raw, side));
    let visited = if mass > 0.0 { weighted / mass } else { 0.0 };
    let mixed = (own + f64::from(visits) * visited) / (f64::from(visits) + 1.0);

    out.clear();
    out.extend(children.map(|child| {
        let stats = tree.stats(child);
        if stats.visits > 0 {
            stats.mean_for(side)
        } else {
            mixed
        }
    }));

    let low = out.iter().copied().fold(f64::INFINITY, f64::min);
    let high = out.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scale = (gumbel.visit_offset + f64::from(most)) * gumbel.value_scale;
    for value in out.iter_mut() {
        *value = scale * (*value - low) / (high - low).max(1e-8);
    }
}

/// The visit count sequential halving considers at each of `simulations` root simulations,
/// for `considered` moves: every considered move once, then the better half more often,
/// round by round, down to two moves (mctx's `get_sequence_of_considered_visits`).
fn considered_visits(considered: usize, simulations: u32) -> Vec<u32> {
    let wanted = usize::try_from(simulations).unwrap_or(usize::MAX);
    if considered <= 1 {
        return (0..simulations).collect();
    }

    let rounds = usize::try_from(usize::BITS - (considered - 1).leading_zeros())
        .expect("a bit count fits in usize");
    let (mut sequence, mut visits) = (Vec::with_capacity(wanted), vec![0u32; considered]);
    let mut kept = considered;
    while sequence.len() < wanted {
        let extra = (wanted / (rounds * kept)).max(1);
        for _ in 0..extra {
            sequence.extend_from_slice(&visits[..kept]);
            for count in &mut visits[..kept] {
                *count += 1;
            }
        }

        kept = (kept / 2).max(2);
    }

    sequence.truncate(wanted);
    sequence
}

/// The log of `child`'s prior, floored at the smallest positive float.
fn log_prior<M: Clone>(tree: &Tree<M>, child: usize) -> f64 {
    libm::log(tree.stats(child).prior.max(f64::MIN_POSITIVE))
}

/// `reward`, [`Side::Max`]'s, as `side` sees it.
fn for_side(reward: f64, side: Side) -> f64 {
    match side {
        Side::Max => reward,
        Side::Min => 1.0 - reward,
    }
}

/// Turns `values` into probabilities, in place: `exp` of each, shifted by the largest, and
/// scaled to sum to one.
fn softmax(values: &mut [f64]) {
    let top = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut total = 0.0;
    for value in values.iter_mut() {
        *value = libm::exp(*value - top);
        total += *value;
    }

    for value in values.iter_mut() {
        *value /= total;
    }
}

/// The item of `items` with the highest `score`, ties going to the earliest.
fn argmax(items: impl Iterator<Item = usize>, score: impl Fn(usize) -> f64) -> Option<usize> {
    items
        .map(|item| (item, score(item)))
        .reduce(|best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        })
        .map(|(item, _)| item)
}

/// A draw from the standard Gumbel distribution: `-ln(-ln(u))` for a uniform `u`.
fn gumbel(rng: &mut impl Random) -> f64 {
    let uniform = unit(rng).max(f64::MIN_POSITIVE);

    -libm::log(-libm::log(uniform))
}

/// Unit tests for the halving schedule, ported from mctx's `seq_halving_test.py`.
#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::considered_visits;

    /// Exactly `considered * log2(considered)` simulations.
    #[test]
    fn the_minimum_simulations_halve_down_to_two() {
        let expected = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
        ];
        assert_eq!(considered_visits(8, 24), expected);
    }

    /// More simulations repeat the rounds, then alternate the last two moves.
    #[test]
    fn extra_simulations_repeat_rounds() {
        let mut expected = vec![0; 8];
        expected.extend([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
        expected.extend((4..17).flat_map(|visits| [visits, visits]));
        expected.push(17);
        assert_eq!(considered_visits(8, 47), expected);
    }

    /// Fewer simulations than one full schedule stop early.
    #[test]
    fn few_simulations_stop_early() {
        assert_eq!(considered_visits(8, 2), [0, 0]);
        let expected = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2];
        assert_eq!(considered_visits(8, 13), expected);
    }

    /// A count that is not a power of two halves by integer division.
    #[test]
    fn odd_counts_halve_down() {
        let expected = [
            0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,
        ];
        assert_eq!(considered_visits(7, 24), expected);
    }

    /// Counts that are not powers of two match mctx's own function: `[0, 0, 0, 1, 1, 2, 2,
    /// 3, 3, 4]` for three moves and ten simulations, from mctx's
    /// `get_sequence_of_considered_visits`.
    #[test]
    fn small_odd_counts_match_mctx() {
        assert_eq!(considered_visits(3, 10), [0, 0, 0, 1, 1, 2, 2, 3, 3, 4]);
        assert_eq!(
            considered_visits(5, 17),
            [0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
        );
        assert_eq!(
            considered_visits(6, 20),
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
        );
    }

    /// With at most one move, every simulation visits it.
    #[test]
    fn one_move_takes_every_simulation() {
        let all: Vec<u32> = (0..16).collect();
        assert_eq!(considered_visits(0, 16), all);
        assert_eq!(considered_visits(1, 16), all);
    }
}

//! The plug points of Monte Carlo tree search: how a child is selected ([`Selection`],
//! default [`Uct`]) and how a new leaf is valued ([`Evaluator`], default [`Rollout`]).

use alloc::vec::Vec;

use super::Rewarded;
use super::random::Random;
use crate::Side;

/// What a search knows about one node of its tree, as selection sees it.
///
/// Non-exhaustive: later selection rules may need more, such as pending evaluations.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct NodeStats {
    /// Simulations that passed through the node; stops counting at `u32::MAX`.
    pub visits: u32,
    /// The sum of [`Side::Max`]'s rewards over those simulations, each in `[0, 1]`.
    pub value: f64,
    /// The node's prior probability among its siblings; uniform unless an evaluator sets
    /// it.
    pub prior: f64,
}

impl NodeStats {
    /// The mean reward for `side`: [`Side::Max`]'s mean, or one minus it for
    /// [`Side::Min`]. `0.5` before the first visit.
    #[must_use]
    pub fn mean_for(&self, side: Side) -> f64 {
        if self.visits == 0 {
            return 0.5;
        }

        let max = self.value / f64::from(self.visits);
        match side {
            Side::Max => max,
            Side::Min => 1.0 - max,
        }
    }
}

/// Chooses which visited child a simulation descends into.
///
/// Children never visited are always tried first, in move order, so a rule only scores
/// visited children; the highest score wins, ties going to the earliest child.
pub trait Selection {
    /// The score of `child` for `side`, who moves at `parent`.
    fn score(&self, side: Side, parent: &NodeStats, child: &NodeStats) -> f64;
}

/// Upper confidence bounds applied to trees (UCT, Kocsis and Szepesvári 2006): the child's
/// mean for the side to move plus `exploration * sqrt(ln(parent visits) / child visits)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Uct {
    /// The exploration constant `c`, finite and non-negative. `sqrt(2)` suits rewards in
    /// `[0, 1]`; larger values explore more.
    pub exploration: f64,
}

impl Default for Uct {
    fn default() -> Self {
        Self {
            exploration: core::f64::consts::SQRT_2,
        }
    }
}

impl Selection for Uct {
    fn score(&self, side: Side, parent: &NodeStats, child: &NodeStats) -> f64 {
        let parent = f64::from(parent.visits.max(1));
        let visits = f64::from(child.visits.max(1));

        child.mean_for(side) + self.exploration * libm::sqrt(libm::log(parent) / visits)
    }
}

/// What an [`Evaluator`] found out about a new leaf.
///
/// Non-exhaustive: later evaluators may also return move priors.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Evaluation {
    /// [`Side::Max`]'s reward from the leaf, in `[0, 1]`.
    pub value: f64,
}

impl Evaluation {
    /// An evaluation of `value`, [`Side::Max`]'s reward in `[0, 1]`.
    #[must_use]
    pub const fn new(value: f64) -> Self {
        Self { value }
    }
}

/// Values a leaf the first time a simulation reaches it.
pub trait Evaluator<G: Rewarded> {
    /// The value of `position`, drawing from `rng` if needed.
    fn evaluate<R: Random>(&mut self, game: &G, position: &G::Position, rng: &mut R) -> Evaluation;
}

/// Random playout: random moves until the game ends or `max_plies` were played, then the
/// [`reward`](Rewarded::reward) of the position reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rollout<M> {
    /// The most moves one playout plays before it scores the position.
    pub max_plies: u32,
    /// The move list, kept for its capacity.
    moves: Vec<M>,
}

impl<M> Rollout<M> {
    /// Playouts of at most `max_plies` moves.
    #[must_use]
    pub const fn new(max_plies: u32) -> Self {
        Self {
            max_plies,
            moves: Vec::new(),
        }
    }
}

impl<M> Default for Rollout<M> {
    /// Playouts of at most 100 moves.
    fn default() -> Self {
        Self::new(100)
    }
}

impl<G: Rewarded> Evaluator<G> for Rollout<G::Move> {
    fn evaluate<R: Random>(&mut self, game: &G, position: &G::Position, rng: &mut R) -> Evaluation {
        let mut moves = core::mem::take(&mut self.moves);
        let mut current = None;
        for _ in 0..self.max_plies {
            let here = current.as_ref().unwrap_or(position);
            moves.clear();
            game.moves(here, &mut moves);
            if moves.is_empty() {
                break;
            }

            let mv = &moves[rng.below(moves.len())];
            current = Some(game.play(here, mv));
        }

        moves.clear();
        self.moves = moves;
        let leaf = current.as_ref().unwrap_or(position);
        Evaluation::new(game.reward(game.evaluate(leaf)))
    }
}

/// The game's own evaluation as the leaf value, without playing on: the
/// [`reward`](Rewarded::reward) of [`evaluate`](crate::Game::evaluate).
///
/// Cheaper than a [`Rollout`] and deterministic; as good as the evaluation is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StaticValue;

impl<G: Rewarded> Evaluator<G> for StaticValue {
    fn evaluate<R: Random>(&mut self, game: &G, position: &G::Position, _: &mut R) -> Evaluation {
        Evaluation::new(game.reward(game.evaluate(position)))
    }
}

/// Unit tests for the node means and UCT scores.
#[cfg(test)]
mod tests {
    use super::{NodeStats, Selection, Uct};
    use crate::Side;

    /// Node statistics with `visits` and `value`.
    const fn stats(visits: u32, value: f64) -> NodeStats {
        NodeStats {
            visits,
            value,
            prior: 1.0,
        }
    }

    /// Means are from the asking side's point of view, and one half before any visit.
    #[test]
    fn means_belong_to_the_side_to_move() {
        assert!((stats(4, 3.0).mean_for(Side::Max) - 0.75).abs() < 1e-12);
        assert!((stats(4, 3.0).mean_for(Side::Min) - 0.25).abs() < 1e-12);
        assert!((stats(0, 0.0).mean_for(Side::Max) - 0.5).abs() < 1e-12);
    }

    /// UCT favours the less visited of two equally good children, and none without
    /// exploration.
    #[test]
    fn uct_explores_the_less_visited_child() {
        let parent = stats(20, 10.0);
        let uct = Uct::default();
        let (rare, common) = (stats(2, 1.0), stats(18, 9.0));
        assert!(uct.score(Side::Max, &parent, &rare) > uct.score(Side::Max, &parent, &common));

        let greedy = Uct { exploration: 0.0 };
        let gap =
            greedy.score(Side::Max, &parent, &rare) - greedy.score(Side::Max, &parent, &common);
        assert!(gap.abs() < 1e-12);
    }
}

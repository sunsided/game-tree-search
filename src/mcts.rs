//! Monte Carlo tree search (MCTS): a tree grown by random simulations instead of a full
//! search to a fixed depth.

mod batch;
#[cfg(feature = "parallel")]
mod parallel;
mod policy;
mod random;
mod tree;

use alloc::vec::Vec;

use search_core::Budget;

pub use policy::{
    BatchEvaluator, BatchLeaf, Evaluation, Evaluator, NodeStats, Puct, Rollout, Selection,
    StaticValue, Uct, Unbatched,
};
#[cfg(feature = "rand_core")]
pub use random::RandCore;
pub use random::{Random, SplitMix64};
use tree::Tree;

use crate::Game;

/// A game whose scores map to rewards, as Monte Carlo tree search averages them.
pub trait Rewarded: Game {
    /// [`Side::Max`](crate::Side::Max)'s reward for `score`, in `[0, 1]`: `1` is the best
    /// outcome for [`Side::Max`](crate::Side::Max), `0` the best for
    /// [`Side::Min`](crate::Side::Min). For a win, a draw and a loss, `1`, `0.5` and `0`.
    fn reward(&self, score: Self::Score) -> f64;
}

/// Counters of one Monte Carlo tree search, which its [`Budget`] sees.
///
/// Non-exhaustive: later versions may add counters. Start from [`TreeStats::default`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct TreeStats {
    /// Simulations run: selection, one leaf evaluation and the backup.
    pub simulations: u64,
    /// Nodes in the tree, the root included; each holds one position's statistics.
    pub nodes: u64,
    /// The depth of the deepest node, counted in moves from the root.
    pub depth: u32,
}

/// A root move with the statistics of the simulations through it.
///
/// Non-exhaustive: later versions may report more per move.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MoveStats<M> {
    /// The move.
    pub mv: M,
    /// Simulations that started with the move.
    pub visits: u32,
    /// The sum of [`Side::Max`](crate::Side::Max)'s rewards over those simulations.
    pub value: f64,
}

impl<M> MoveStats<M> {
    /// [`Side::Max`](crate::Side::Max)'s mean reward after the move; `0.5` if unvisited.
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.visits == 0 {
            0.5
        } else {
            self.value / f64::from(self.visits)
        }
    }
}

/// The result of a Monte Carlo tree search.
///
/// Non-exhaustive: later versions may report more.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MctsResult<M> {
    /// The move with the most visits, ties going to the earliest; with no simulations, the
    /// first move. `None` if the root has no moves.
    pub best: Option<M>,
    /// Every root move in move order, with its visits and value.
    ///
    /// The visit counts divided by [`TreeStats::simulations`] are the search's move
    /// distribution, and the stats of several searches, for example over sampled deals of
    /// hidden cards, can be added up move by move.
    pub root: Vec<MoveStats<M>>,
    /// The most visited line: from the root, the most visited child of each node, while it
    /// was visited.
    pub pv: Vec<M>,
    /// What the search did.
    pub stats: TreeStats,
}

/// Monte Carlo tree search with replaceable selection, leaf evaluation and random source.
///
/// Each simulation descends from the root, choosing unvisited children first and then by
/// the [`Selection`] rule for the side to move at each node, so one side may move several
/// times in a row. It adds the first unvisited node it reaches, values it with the
/// [`Evaluator`] and adds the reward to every node on the way (mean backup). A position
/// without moves is valued by its [`Rewarded::reward`].
///
/// The defaults are [`Uct`] selection, [`Rollout`] playouts and a [`SplitMix64`] generator
/// seeded with `0`; with the `rand_core` feature, `RandCore` wraps any `rand_core`
/// generator. With the `parallel` feature, `Mcts::search_parallel` runs several trees on
/// `std` threads and adds up their root statistics. The generator continues across searches, so a searcher reproduces the
/// same sequence of searches from the same seed. The tree is rebuilt for every search, in
/// buffers kept between searches. Chance positions of [`Stochastic`](crate::Stochastic) games are not sampled:
/// MCTS sees only [`Game::moves`].
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
/// use game_tree_search::{Mcts, Rewarded, TreeStats};
///
/// impl Rewarded for Nim {
///     fn reward(&self, score: i32) -> f64 {
///         f64::from(score + 1) / 2.0
///     }
/// }
///
/// // Seven stones: taking one leaves six, a lost pile for the opponent.
/// let mut mcts = Mcts::new();
/// let result = mcts.search(&Nim, &(7, Side::Max), |stats: &TreeStats| stats.simulations >= 2000);
/// assert_eq!(result.best, Some(1));
/// assert_eq!(result.root.iter().map(|root| root.visits).sum::<u32>(), 2000);
/// ```
#[derive(Debug)]
pub struct Mcts<G: Game, E = Rollout<<G as Game>::Move>, S = Uct, R = SplitMix64> {
    /// Values new leaves.
    evaluator: E,
    /// Chooses among visited children.
    selection: S,
    /// Draws the random numbers.
    rng: R,
    /// The search tree, kept for its capacity.
    tree: Tree<G::Move>,
}

impl<G: Rewarded> Mcts<G> {
    /// A search with [`Uct`] selection, [`Rollout`] playouts of up to 100 moves and a
    /// [`SplitMix64`] generator seeded with `0`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            evaluator: Rollout::default(),
            selection: Uct::default(),
            rng: SplitMix64::new(0),
            tree: Tree::new(),
        }
    }
}

impl<G: Rewarded> Default for Mcts<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: Rewarded, E: Evaluator<G>, S: Selection, R: Random> Mcts<G, E, S, R> {
    /// This search, valuing new leaves with `evaluator`.
    #[must_use]
    pub fn with_evaluator<E2: Evaluator<G>>(self, evaluator: E2) -> Mcts<G, E2, S, R> {
        Mcts {
            evaluator,
            selection: self.selection,
            rng: self.rng,
            tree: self.tree,
        }
    }

    /// This search, choosing children with `selection`.
    #[must_use]
    pub fn with_selection<S2: Selection>(self, selection: S2) -> Mcts<G, E, S2, R> {
        Mcts {
            evaluator: self.evaluator,
            selection,
            rng: self.rng,
            tree: self.tree,
        }
    }

    /// This search, drawing random numbers from `rng`.
    #[must_use]
    pub fn with_rng<R2: Random>(self, rng: R2) -> Mcts<G, E, S, R2> {
        Mcts {
            evaluator: self.evaluator,
            selection: self.selection,
            rng,
            tree: self.tree,
        }
    }

    /// Searches `position`, running simulations until `budget` is exhausted.
    ///
    /// The budget is asked before every simulation, so a limit of `n` simulations runs
    /// exactly `n`. Bound memory by also limiting [`TreeStats::nodes`].
    ///
    /// After [`Mcts::advance`] kept a subtree, the search continues from it if `position`
    /// fits it: the same side to move and the same moves in the same order. Otherwise, for
    /// example for the root of a new game, it starts a new tree.
    /// [`TreeStats::simulations`] counts this search's simulations only, while the root
    /// moves' visits include the kept ones.
    pub fn search(
        &mut self,
        game: &G,
        position: &G::Position,
        mut budget: impl Budget<TreeStats>,
    ) -> MctsResult<G::Move> {
        self.tree.begin(game, position, &mut self.evaluator);
        while !budget.exhausted(&self.tree.stats) {
            self.simulate(game, position);
        }

        self.tree.result()
    }

    /// Keeps the subtree under `moves`, played from the last search's root, for the next
    /// search to continue from (tree reuse), and drops the rest of the tree.
    ///
    /// Typically the move the search chose and the opponent's reply. Returns whether a
    /// subtree was kept: not if a move is not in the tree, or if the position it leads to was
    /// never visited; the next search then starts a new tree, as without this call. Only
    /// [`Mcts::search`] continues a kept tree; with the `parallel` feature,
    /// `Mcts::search_parallel` always starts new trees.
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
    /// use game_tree_search::{Mcts, TreeStats};
    ///
    /// let budget = |stats: &TreeStats| stats.simulations >= 1000;
    /// let mut mcts = Mcts::new();
    /// let first = mcts.search(&Nim, &(10, Side::Max), budget);
    /// assert_eq!(first.best, Some(1));
    ///
    /// // Max takes 1, Min takes 2: keep what the search learned about the 7 stones left.
    /// assert!(mcts.advance(&[1, 2]));
    /// let second = mcts.search(&Nim, &(7, Side::Max), budget);
    /// let kept: u32 = second.root.iter().map(|root| root.visits).sum();
    /// assert!(kept > 1000);
    /// assert_eq!(second.best, Some(1));
    /// ```
    pub fn advance(&mut self, moves: &[G::Move]) -> bool {
        self.tree.advance(moves)
    }

    /// One simulation from the root `position`: select, expand, evaluate, back up.
    fn simulate(&mut self, game: &G, position: &G::Position) {
        let leaf = self
            .tree
            .select(game, position, (&self.selection, &mut self.evaluator));
        let reached = leaf.position.as_ref().unwrap_or(position);
        let value = if leaf.finished {
            game.reward(game.evaluate(reached))
        } else {
            self.evaluator.evaluate(game, reached, &mut self.rng).value
        };

        self.tree.backup(value);
    }
}

//! Root-parallel Monte Carlo tree search (feature `parallel`): independent trees on `std`
//! threads, their root statistics added up.

use alloc::vec::Vec;
use std::panic;
use std::thread;

use search_core::Budget;

use super::policy::{Evaluator, Selection};
use super::random::{Random, SplitMix64};
use super::tree::Tree;
use super::{Mcts, MctsResult, Rewarded, TreeStats};

impl<G, E, S> Mcts<G, E, S, SplitMix64>
where
    G: Rewarded + Sync,
    G::Position: Sync,
    G::Move: Send,
    E: Evaluator<G> + Clone + Send,
    S: Selection + Clone + Send,
{
    /// Searches `position` with `threads` independent trees in parallel (root parallelism),
    /// each spending its own copy of `budget`, and adds up their root statistics.
    ///
    /// Each thread gets a copy of the evaluator and selection rule and a [`SplitMix64`]
    /// seeded from this searcher's generator, so the same seed reproduces the same merged
    /// result. The trees share nothing, so the merged visits are `threads` times one
    /// search's; the best move is the most visited one over all trees, ties going to the
    /// earliest. The line is that of the tree that visited the best move most. A `threads`
    /// of `0` runs one tree. The game's moves and their order must not depend on anything
    /// but the position, so every tree lists the root moves in the same order.
    ///
    /// Available with the `parallel` feature, which needs `std`.
    ///
    /// # Panics
    ///
    /// Panics if a thread panics, for example in the game's own code; the first such panic
    /// is passed on.
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
    /// let budget = |stats: &TreeStats| stats.simulations >= 500;
    /// let result = Mcts::new().search_parallel(&Nim, &(7, Side::Max), 4, budget);
    /// assert_eq!(result.best, Some(1));
    /// assert_eq!(result.stats.simulations, 2000);
    /// ```
    pub fn search_parallel<B: Budget<TreeStats> + Clone + Send>(
        &mut self,
        game: &G,
        position: &G::Position,
        threads: usize,
        budget: B,
    ) -> MctsResult<G::Move> {
        let threads = threads.max(1);
        let mut budgets: Vec<B> = (1..threads).map(|_| budget.clone()).collect();
        budgets.push(budget);
        let workers: Vec<_> = budgets
            .into_iter()
            .map(|budget| (self.worker(), budget))
            .collect();
        let results = thread::scope(|scope| {
            let workers: Vec<_> = workers
                .into_iter()
                .map(|(mut worker, budget)| {
                    scope.spawn(move || worker.search(game, position, budget))
                })
                .collect();

            workers
                .into_iter()
                .map(|worker| {
                    worker
                        .join()
                        .unwrap_or_else(|panic| panic::resume_unwind(panic))
                })
                .collect::<Vec<_>>()
        });

        merge(&results)
    }

    /// A searcher for one thread: copies of this one's evaluator and selection rule, its
    /// own generator seeded from this one's, and a fresh tree.
    fn worker(&mut self) -> Self {
        Self {
            evaluator: self.evaluator.clone(),
            selection: self.selection.clone(),
            rng: SplitMix64::new(self.rng.next_u64()),
            tree: Tree::new(),
        }
    }
}

/// The results of independent searches of one root, added up move by move.
///
/// Every result lists the same root moves in the same order. The best move is the most
/// visited over all of them, ties going to the earliest; the line is that of the result
/// that visited it most, ties going to the earliest result.
fn merge<M: Clone>(results: &[MctsResult<M>]) -> MctsResult<M> {
    let mut merged = results[0].clone();
    for result in &results[1..] {
        for (total, root) in merged.root.iter_mut().zip(&result.root) {
            total.visits = total.visits.saturating_add(root.visits);
            total.value += root.value;
        }

        merged.stats.simulations += result.stats.simulations;
        merged.stats.nodes += result.stats.nodes;
        merged.stats.depth = merged.stats.depth.max(result.stats.depth);
    }

    let Some(best) = most_visited(&merged) else {
        return merged;
    };

    let visits = |result: &&MctsResult<M>| result.root[best].visits;
    let deepest = results.iter().reduce(|top, result| {
        if visits(&result) > visits(&top) {
            result
        } else {
            top
        }
    });
    merged.pv = deepest.map_or_else(Vec::new, |result| result.pv.clone());
    merged.best = Some(merged.root[best].mv.clone());

    merged
}

/// The index of the most visited root move, ties going to the earliest; `None` without
/// root moves.
fn most_visited<M>(result: &MctsResult<M>) -> Option<usize> {
    let visits = |index: usize| result.root[index].visits;

    (0..result.root.len()).reduce(|best, index| {
        if visits(index) > visits(best) {
            index
        } else {
            best
        }
    })
}

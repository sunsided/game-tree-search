//! Tree-parallel Monte Carlo tree search (feature `parallel`): several threads grow one
//! tree, selecting and backing up under one lock and evaluating outside it, with virtual
//! loss spreading them over different lines (Chaslot et al. 2008, global mutex).

use alloc::vec::Vec;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::thread;

use search_core::Budget;

use super::policy::{Evaluator, Selection};
use super::random::{Random, SplitMix64};
use super::tree::Tree;
use super::{Mcts, MctsResult, Rewarded, TreeStats};
use crate::lazy_smp::rethrow;

/// The tree and what the threads share besides it, behind one lock.
struct Shared<M, B> {
    /// The tree.
    tree: Tree<M>,
    /// The budget, asked before every simulation starts.
    budget: B,
    /// Simulations started and not yet backed up, counted against the budget.
    running: u64,
}

impl<G, E, S> Mcts<G, E, S, SplitMix64>
where
    G: Rewarded + Sync,
    G::Position: Sync,
    G::Move: Send,
    E: Evaluator<G> + Clone + Send,
    S: Selection + Sync,
{
    /// Searches `position` with `threads` threads growing one shared tree (tree
    /// parallelism), and returns its result.
    ///
    /// Each thread repeats a simulation: under the tree's lock, it descends to a new leaf
    /// and marks its path as waiting, so the other threads' selection counts it as a loss
    /// (virtual loss) and spreads out; outside the lock, it values the leaf with its own copy
    /// of the evaluator and its own [`SplitMix64`], seeded from this searcher's; under the
    /// lock again, it backs the value up. A finished game is valued by its reward under the
    /// lock. The budget is asked before every simulation with the running ones counted, so
    /// a limit of `n` simulations still runs exactly `n`.
    ///
    /// The threads share one tree, so the result is that of one search with more
    /// simulations, not a sum of trees as in [`Mcts::search_parallel`]. It pays off only
    /// when valuing a leaf costs far more than selecting it, as a network evaluation does.
    /// In the `parallel` benchmarks, with leaves of about a microsecond, every added thread
    /// made it slower than one thread, since the threads queue for the lock, while root
    /// parallelism ran 2.4 times faster on eight threads. With evaluations of about
    /// 30 microseconds, eight threads finished 2,000 simulations in 16 ms on the shared tree,
    /// against 26 ms with root parallelism and 60 ms on one thread. Which thread reaches which leaf first depends on scheduling, so unlike
    /// [`Mcts::search_parallel`], the same seed does not reproduce the same result. A
    /// `threads` of `0` runs one thread. Continues a tree kept by [`Mcts::advance`] like
    /// [`Mcts::search`], and adds the [root noise](Mcts::with_root_noise) to a new one.
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
    /// let budget = |stats: &TreeStats| stats.simulations >= 2000;
    /// let result = Mcts::new().search_shared(&Nim, &(7, Side::Max), 4, budget);
    /// assert_eq!(result.best, Some(1));
    /// assert_eq!(result.stats.simulations, 2000);
    /// ```
    pub fn search_shared<B: Budget<TreeStats> + Send>(
        &mut self,
        game: &G,
        position: &G::Position,
        threads: usize,
        budget: B,
    ) -> MctsResult<G::Move> {
        self.tree.begin(game, position, &mut self.evaluator);
        self.add_root_noise();
        let workers: Vec<_> = (0..threads.max(1))
            .map(|_| {
                let rng = SplitMix64::new(self.rng.next_u64());
                (self.evaluator.clone(), rng)
            })
            .collect();
        let shared = Mutex::new(Shared {
            tree: core::mem::replace(&mut self.tree, Tree::new()),
            budget,
            running: 0,
        });
        let selection = &self.selection;

        thread::scope(|scope| {
            let handles: Vec<_> = workers
                .into_iter()
                .map(|(mut evaluator, mut rng)| {
                    let shared = &shared;
                    scope.spawn(move || {
                        simulate_until_spent(
                            game,
                            position,
                            shared,
                            (selection, &mut evaluator),
                            &mut rng,
                        );
                    })
                })
                .collect();
            for handle in handles {
                rethrow(handle.join());
            }
        });

        self.tree = shared
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
            .tree;
        self.tree.result()
    }
}

/// One thread's share of [`Mcts::search_shared`]: simulations until the budget is spent.
fn simulate_until_spent<G, B>(
    game: &G,
    position: &G::Position,
    shared: &Mutex<Shared<G::Move, B>>,
    (selection, evaluator): (&impl Selection, &mut impl Evaluator<G>),
    rng: &mut impl Random,
) where
    G: Rewarded,
    B: Budget<TreeStats>,
{
    let mut path = Vec::new();
    loop {
        let mut state = lock(shared);
        let counted = TreeStats {
            simulations: state.tree.stats.simulations + state.running,
            ..state.tree.stats
        };
        if state.budget.exhausted(&counted) {
            return;
        }

        let leaf = state.tree.select(game, position, (selection, evaluator));
        let reached = match leaf.position {
            Some(reached) if !leaf.finished => reached,
            reached => {
                let here = reached.as_ref().unwrap_or(position);
                state.tree.backup(game.reward(game.evaluate(here)));
                continue;
            }
        };

        path.clear();
        path.extend_from_slice(state.tree.path());
        state.tree.pend(&path, true);
        state.running += 1;
        drop(state);

        let value = evaluator.evaluate(game, &reached, rng).value;
        let mut state = lock(shared);
        state.tree.pend(&path, false);
        state.tree.backup_path(&path, value);
        state.running -= 1;
    }
}

/// Locks `shared`, ignoring a poisoned lock: a thread that panicked left at worst one path
/// marked as waiting, and its panic ends the search anyway.
fn lock<T>(shared: &Mutex<T>) -> MutexGuard<'_, T> {
    shared.lock().unwrap_or_else(PoisonError::into_inner)
}

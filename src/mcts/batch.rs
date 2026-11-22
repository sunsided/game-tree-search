//! Batched leaf evaluation for Monte Carlo tree search: new leaves wait under virtual loss
//! until a [`BatchEvaluator`] values them together.

use alloc::vec::Vec;

use search_core::Budget;

use super::policy::{BatchEvaluator, BatchLeaf, Evaluator, Selection};
use super::random::Random;
use super::{Mcts, MctsResult, Rewarded, TreeStats};

/// The leaves of one batch, in buffers kept for their capacity.
struct Batch<P, M> {
    /// The path from the root to each leaf, flattened; `paths[ends[i - 1]..ends[i]]`.
    paths: Vec<usize>,
    /// Where each leaf's path ends in `paths`.
    ends: Vec<usize>,
    /// Each leaf's position.
    positions: Vec<P>,
    /// The moves of every leaf, flattened like the paths; `moves[cuts[i - 1]..cuts[i]]`.
    moves: Vec<M>,
    /// Where each leaf's moves end in `moves`.
    cuts: Vec<usize>,
}

impl<P, M> Batch<P, M> {
    /// The `i`-th leaf's path and moves.
    fn leaf(&self, i: usize) -> (&[usize], &[M]) {
        let (from, to) = (if i == 0 { 0 } else { self.ends[i - 1] }, self.ends[i]);
        let (first, last) = (if i == 0 { 0 } else { self.cuts[i - 1] }, self.cuts[i]);

        (&self.paths[from..to], &self.moves[first..last])
    }
}

impl<G: Rewarded, E: Evaluator<G>, S: Selection, R: Random> Mcts<G, E, S, R> {
    /// Searches `position` like [`Mcts::search`], valuing new leaves `batch` at a time with
    /// `evaluator` instead of one by one.
    ///
    /// A simulation stops at its new leaf and waits; until the batch is valued, selection
    /// counts every waiting simulation as a loss for the side choosing it (virtual loss), so
    /// the next simulations of the batch spread over other moves. A finished game is valued
    /// by its reward at once. Then `evaluator` values the batch in one call, its values are
    /// backed up and its priors, if any, set on the leaves' moves. A `batch` of `0` or `1`
    /// values every leaf on its own, like [`Mcts::search`] with that evaluator.
    ///
    /// On a new tree, the root goes to `evaluator` first, as a batch of its own, for the
    /// priors of its moves; that call is not a simulation. The budget is asked before every
    /// simulation, with the waiting ones counted, so a limit of `n` simulations still runs
    /// exactly `n`. The searcher's own [`Evaluator`]
    /// gives the priors of new nodes until the batch sets them. Continues a tree kept by
    /// [`Mcts::advance`] like [`Mcts::search`].
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
    /// use game_tree_search::{Mcts, Rollout, TreeStats, Unbatched};
    ///
    /// // Rollouts, eight leaves per call: a stand-in for a network evaluating batches.
    /// let mut evaluator = Unbatched::new(Rollout::default(), 7);
    /// let budget = |stats: &TreeStats| stats.simulations >= 2000;
    /// let result = Mcts::new().search_batched(&Nim, &(7, Side::Max), 8, &mut evaluator, budget);
    /// assert_eq!(result.best, Some(1));
    /// assert_eq!(result.stats.simulations, 2000);
    /// ```
    pub fn search_batched(
        &mut self,
        game: &G,
        position: &G::Position,
        batch: usize,
        evaluator: &mut impl BatchEvaluator<G>,
        mut budget: impl Budget<TreeStats>,
    ) -> MctsResult<G::Move> {
        self.tree.begin(game, position, &mut self.evaluator);
        if self.tree.fresh() {
            self.root_priors(game, position, evaluator);
        }

        self.add_root_noise();

        let mut leaves = Batch {
            paths: Vec::new(),
            ends: Vec::new(),
            positions: Vec::new(),
            moves: Vec::new(),
            cuts: Vec::new(),
        };
        loop {
            let full = self.fill(game, position, batch.max(1), (&mut leaves, &mut budget));
            if !leaves.ends.is_empty() {
                self.value(game, evaluator, &mut leaves);
            }

            if !full {
                break;
            }
        }

        self.tree.result()
    }

    /// Asks `evaluator` for the priors of the root's moves, as `AlphaZero` evaluates the root
    /// before its first simulation; the value is not used.
    fn root_priors(
        &mut self,
        game: &G,
        position: &G::Position,
        evaluator: &mut impl BatchEvaluator<G>,
    ) {
        let mut moves = Vec::new();
        self.tree.moves_of(0, &mut moves);
        if moves.is_empty() {
            return;
        }

        let root = [BatchLeaf {
            position,
            moves: &moves,
        }];
        let (mut values, mut priors) = (Vec::new(), Vec::new());
        evaluator.evaluate_batch(game, &root, &mut values, &mut priors);
        if priors.len() == moves.len() {
            self.tree.set_priors(0, &priors);
        }
    }

    /// Selects new leaves into `leaves` until it holds `batch` or `budget` is spent, backing
    /// up finished games at once. Returns whether the batch filled up, so more may follow.
    fn fill<B: Budget<TreeStats>>(
        &mut self,
        game: &G,
        position: &G::Position,
        batch: usize,
        (leaves, budget): (&mut Batch<G::Position, G::Move>, &mut B),
    ) -> bool {
        while leaves.ends.len() < batch {
            let waiting = u64::try_from(leaves.ends.len()).unwrap_or(u64::MAX);
            let stats = TreeStats {
                simulations: self.tree.stats.simulations + waiting,
                ..self.tree.stats
            };
            if budget.exhausted(&stats) {
                return false;
            }

            let leaf = self
                .tree
                .select(game, position, (&self.selection, &mut self.evaluator));
            match leaf.position {
                Some(reached) if !leaf.finished => {
                    let start = leaves.paths.len();
                    leaves.paths.extend_from_slice(self.tree.path());
                    let path = &leaves.paths[start..];
                    self.tree.pend(path, true);
                    let node = *path.last().expect("a path has a node");
                    self.tree.moves_of(node, &mut leaves.moves);
                    leaves.ends.push(leaves.paths.len());
                    leaves.cuts.push(leaves.moves.len());
                    leaves.positions.push(reached);
                }
                reached => {
                    let here = reached.as_ref().unwrap_or(position);
                    self.tree.backup(game.reward(game.evaluate(here)));
                }
            }
        }

        true
    }

    /// Values the waiting `leaves`, at least one, with `evaluator`, backs them up and empties
    /// the batch.
    fn value(
        &mut self,
        game: &G,
        evaluator: &mut impl BatchEvaluator<G>,
        leaves: &mut Batch<G::Position, G::Move>,
    ) {
        let requests: Vec<_> = (0..leaves.ends.len())
            .map(|i| BatchLeaf {
                position: &leaves.positions[i],
                moves: leaves.leaf(i).1,
            })
            .collect();
        let (mut values, mut priors) = (Vec::new(), Vec::new());
        evaluator.evaluate_batch(game, &requests, &mut values, &mut priors);
        drop(requests);

        let all_priors = priors.len() == leaves.moves.len();
        for i in 0..leaves.ends.len() {
            let (path, moves) = leaves.leaf(i);
            let node = *path.last().expect("a path has a node");
            if all_priors {
                let first = if i == 0 { 0 } else { leaves.cuts[i - 1] };
                self.tree
                    .set_priors(node, &priors[first..first + moves.len()]);
            }

            self.tree.pend(path, false);
            self.tree
                .backup_path(path, values.get(i).copied().unwrap_or(0.5));
        }

        leaves.paths.clear();
        leaves.ends.clear();
        leaves.positions.clear();
        leaves.moves.clear();
        leaves.cuts.clear();
    }
}

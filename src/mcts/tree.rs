//! The search tree of Monte Carlo tree search: nodes in one arena, children stored next to
//! each other, positions replayed from the root rather than stored. Every node is expanded
//! when it is added to a simulation's path, the root when the search starts.

use alloc::vec::Vec;

use super::Rewarded;
use super::policy::{Evaluator, NodeStats, Selection};
use super::{MctsResult, MoveStats, TreeStats};
use crate::Side;

/// One position of the tree.
#[derive(Debug, Clone)]
struct Node<M> {
    /// The move that leads here; `None` at the root.
    mv: Option<M>,
    /// The index of the first child in the arena; the children follow it.
    first: usize,
    /// The number of children; `0` until expanded, or where the game ends.
    children: usize,
    /// What the simulations through the node found.
    stats: NodeStats,
    /// The side to move; `None` until expanded, or where the game ends.
    side: Option<Side>,
    /// Moves from the root to the node.
    depth: u32,
    /// Simulations through the node waiting for a batched evaluation; selection counts each
    /// as a loss for the side choosing the node (virtual loss).
    pending: u32,
}

/// Where a simulation's descent ended.
pub(super) struct Leaf<P> {
    /// The position reached; `None` if the descent stayed at the root.
    pub(super) position: Option<P>,
    /// Whether the game ends there, so its reward needs no evaluation.
    pub(super) finished: bool,
}

/// The tree of one search, with buffers kept between searches.
#[derive(Debug, Clone)]
pub(super) struct Tree<M> {
    /// Every node; the root is at index `0`.
    nodes: Vec<Node<M>>,
    /// The nodes the current simulation passed through, from the root.
    path: Vec<usize>,
    /// The move list for expansions, kept for its capacity.
    moves: Vec<M>,
    /// The prior list for expansions, kept for its capacity.
    priors: Vec<f64>,
    /// The arena a re-rooted tree is copied into, kept for its capacity.
    spare: Vec<Node<M>>,
    /// Whether [`Tree::advance`] kept a subtree for the next search to continue.
    pub(super) kept: bool,
    /// The counters of the current search.
    pub(super) stats: TreeStats,
}

impl<M: Clone> Tree<M> {
    /// An empty tree.
    pub(super) const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            path: Vec::new(),
            moves: Vec::new(),
            priors: Vec::new(),
            spare: Vec::new(),
            kept: false,
            stats: TreeStats {
                simulations: 0,
                nodes: 0,
                depth: 0,
            },
        }
    }

    /// Starts a new search at `root`, with the root expanded so its moves are listed even
    /// before the first simulation.
    pub(super) fn reset<G: Rewarded<Move = M>>(
        &mut self,
        game: &G,
        root: &G::Position,
        evaluator: &mut impl Evaluator<G>,
    ) {
        self.nodes.clear();
        self.stats = TreeStats::default();
        self.nodes.push(Node {
            mv: None,
            first: 0,
            children: 0,
            stats: NodeStats {
                visits: 0,
                value: 0.0,
                prior: 1.0,
            },
            side: None,
            depth: 0,
            pending: 0,
        });
        self.expand(game, root, 0, evaluator);
    }

    /// Starts the next search: on the subtree [`Tree::advance`] kept, counting simulations
    /// from zero again, if it still fits `root`; else on a new tree at `root`.
    pub(super) fn begin<G: Rewarded<Move = M>>(
        &mut self,
        game: &G,
        root: &G::Position,
        evaluator: &mut impl Evaluator<G>,
    ) where
        M: PartialEq,
    {
        let kept = core::mem::replace(&mut self.kept, false);
        if kept && self.fits(game, root) {
            self.stats.simulations = 0;
        } else {
            self.reset(game, root, evaluator);
        }
    }

    /// Whether the kept root could be `root`: the same side to move and the same moves in
    /// the same order.
    // shortcut: positions with the same side and moves are not told apart; compare a
    // position key if games ever need that.
    fn fits<G: Rewarded<Move = M>>(&mut self, game: &G, root: &G::Position) -> bool
    where
        M: PartialEq,
    {
        let mut moves = core::mem::take(&mut self.moves);
        moves.clear();
        game.moves(root, &mut moves);
        game.order_moves(root, &mut moves);

        let node = &self.nodes[0];
        let children = &self.nodes[node.first..node.first + node.children];
        let side = (!moves.is_empty()).then(|| game.side_to_move(root));
        let same = node.side == side
            && children.len() == moves.len()
            && children
                .iter()
                .zip(&moves)
                .all(|(child, mv)| child.mv.as_ref() == Some(mv));
        self.moves = moves;

        same
    }

    /// Re-roots the tree at the node `moves` lead to from the root, dropping everything
    /// else, so the next search continues from its statistics. Returns whether a subtree was
    /// kept; it is not if a move is not in the tree or the node was never visited.
    pub(super) fn advance(&mut self, moves: &[M]) -> bool
    where
        M: PartialEq,
    {
        let target = self.descend(moves).filter(|&index| {
            self.nodes
                .get(index)
                .is_some_and(|node| node.stats.visits > 0)
        });
        self.kept = target.is_some();
        if let Some(target) = target {
            self.reroot(target);
        }

        self.kept
    }

    /// The node `moves` lead to from the root, if every move is in the tree.
    fn descend(&self, moves: &[M]) -> Option<usize>
    where
        M: PartialEq,
    {
        moves.iter().try_fold(0, |index, mv| {
            let node = self.nodes.get(index)?;
            (node.first..node.first + node.children)
                .find(|&child| self.nodes[child].mv.as_ref() == Some(mv))
        })
    }

    /// Makes `target` the root: copies its subtree into the spare arena, children still next
    /// to each other, and swaps the arenas.
    fn reroot(&mut self, target: usize) {
        let mut kept = core::mem::take(&mut self.spare);
        kept.clear();
        let offset = self.nodes[target].depth;
        kept.push(Node {
            mv: None,
            ..self.nodes[target].clone()
        });

        let mut next = 0;
        while next < kept.len() {
            let (first, children) = (kept[next].first, kept[next].children);
            kept[next].first = kept.len();
            kept.extend_from_slice(&self.nodes[first..first + children]);

            next += 1;
        }

        for node in &mut kept {
            node.depth -= offset;
        }

        self.spare = core::mem::replace(&mut self.nodes, kept);
        self.stats.nodes = u64::try_from(self.nodes.len()).unwrap_or(u64::MAX);
        self.stats.depth = self
            .nodes
            .iter()
            .filter(|node| node.children > 0)
            .map(|node| node.depth + 1)
            .max()
            .unwrap_or(0);
    }

    /// Descends from the root: through visited children chosen by `selection`, into the
    /// first unvisited child, which it expands to tell whether the game ends there, or to a
    /// node where the game ends. Records the path.
    pub(super) fn select<G: Rewarded<Move = M>>(
        &mut self,
        game: &G,
        root: &G::Position,
        (selection, evaluator): (&impl Selection, &mut impl Evaluator<G>),
    ) -> Leaf<G::Position> {
        self.path.clear();
        self.path.push(0);
        let (mut index, mut current) = (0, None);
        loop {
            let here = current.as_ref().unwrap_or(root);
            let Some(child) = self.choose(index, selection) else {
                return Leaf {
                    position: current,
                    finished: true,
                };
            };

            let mv = self.nodes[child].mv.as_ref().expect("a child has a move");
            let next = game.play(here, mv);
            self.path.push(child);
            if self.untouched(child) {
                self.expand(game, &next, child, evaluator);
                return Leaf {
                    finished: self.nodes[child].children == 0,
                    position: Some(next),
                };
            }

            (index, current) = (child, Some(next));
        }
    }

    /// The child of `index` a simulation descends into: the first unvisited one if
    /// `selection` asks for that, else the best by `selection` for the side to move, ties
    /// going to the earliest. `None` if the game ends at `index`.
    fn choose(&self, index: usize, selection: &impl Selection) -> Option<usize> {
        let node = &self.nodes[index];
        let side = node.side?;
        let children = node.first..node.first + node.children;
        let unvisited = selection
            .unvisited_first()
            .then(|| children.clone().find(|&child| self.untouched(child)));
        if let Some(unvisited) = unvisited.flatten() {
            return Some(unvisited);
        }

        let parent = self.seen(index, side);
        let scored = children.map(|child| {
            let score = selection.score(side, &parent, &self.seen(child, side));
            (child, score)
        });
        let best = scored.reduce(|best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        });

        best.map(|(child, _)| child)
    }

    /// Adds the children of `index`, the moves of `position`, in [`Game::order_moves`](crate::Game::order_moves) order,
    /// with the priors `evaluator` gives them.
    fn expand<G: Rewarded<Move = M>>(
        &mut self,
        game: &G,
        position: &G::Position,
        index: usize,
        evaluator: &mut impl Evaluator<G>,
    ) {
        let mut moves = core::mem::take(&mut self.moves);
        moves.clear();
        game.moves(position, &mut moves);
        game.order_moves(position, &mut moves);
        self.priors.clear();
        if !moves.is_empty() {
            evaluator.priors(game, position, &moves, &mut self.priors);
        }

        normalize(&mut self.priors, moves.len());
        let (first, children) = (self.nodes.len(), moves.len());
        let depth = self.nodes[index].depth + 1;
        for (mv, &prior) in moves.drain(..).zip(&self.priors) {
            self.nodes.push(Node {
                mv: Some(mv),
                first: 0,
                children: 0,
                stats: NodeStats {
                    visits: 0,
                    value: 0.0,
                    prior,
                },
                side: None,
                depth,
                pending: 0,
            });
        }

        self.moves = moves;
        let side = (children > 0).then(|| game.side_to_move(position));
        let node = &mut self.nodes[index];
        (node.first, node.children, node.side) = (first, children, side);
        self.stats.nodes = u64::try_from(self.nodes.len()).unwrap_or(u64::MAX);
        if children > 0 {
            self.stats.depth = self.stats.depth.max(depth);
        }
    }

    /// Adds `value`, [`Side::Max`]'s reward, to every node of the last simulation's path.
    pub(super) fn backup(&mut self, value: f64) {
        let path = core::mem::take(&mut self.path);
        self.backup_path(&path, value);
        self.path = path;
    }

    /// Adds `value`, [`Side::Max`]'s reward, to every node of `path`, one simulation.
    pub(super) fn backup_path(&mut self, path: &[usize], value: f64) {
        for &index in path {
            let stats = &mut self.nodes[index].stats;
            stats.visits = stats.visits.saturating_add(1);
            stats.value += value;
        }

        self.stats.simulations += 1;
    }

    /// The nodes the last simulation passed through, from the root.
    pub(super) fn path(&self) -> &[usize] {
        &self.path
    }

    /// Marks every node of `path` as waiting for one more evaluation, or one fewer.
    pub(super) fn pend(&mut self, path: &[usize], waiting: bool) {
        for &index in path {
            let pending = &mut self.nodes[index].pending;
            *pending = if waiting {
                pending.saturating_add(1)
            } else {
                pending.saturating_sub(1)
            };
        }
    }

    /// The moves of `index`'s children, in order, appended to `moves`.
    pub(super) fn moves_of(&self, index: usize, moves: &mut Vec<M>) {
        let node = &self.nodes[index];
        let children = &self.nodes[node.first..node.first + node.children];
        moves.extend(children.iter().filter_map(|child| child.mv.clone()));
    }

    /// Replaces the priors of `index`'s children with `priors`, one per child, scaled to sum
    /// to one, or uniform if they are invalid.
    pub(super) fn set_priors(&mut self, index: usize, priors: &[f64]) {
        let (first, children) = (self.nodes[index].first, self.nodes[index].children);
        let mut scaled = core::mem::take(&mut self.priors);
        scaled.clear();
        scaled.extend_from_slice(priors);
        normalize(&mut scaled, children);
        for (child, &prior) in self.nodes[first..first + children].iter_mut().zip(&scaled) {
            child.stats.prior = prior;
        }

        self.priors = scaled;
    }

    /// Whether the root has not been visited yet, so its move priors are still those of its
    /// expansion.
    pub(super) fn fresh(&self) -> bool {
        self.nodes
            .first()
            .is_some_and(|root| root.stats.visits == 0)
    }

    /// Whether no simulation has reached `index` yet, finished or waiting.
    fn untouched(&self, index: usize) -> bool {
        self.nodes[index].stats.visits == 0 && self.nodes[index].pending == 0
    }

    /// The statistics of `index` as selection for `side` sees them: every waiting
    /// simulation counted as a visit lost for `side` (virtual loss).
    fn seen(&self, index: usize, side: Side) -> NodeStats {
        let node = &self.nodes[index];
        let lost = match side {
            Side::Max => 0.0,
            Side::Min => 1.0,
        };

        NodeStats {
            visits: node.stats.visits.saturating_add(node.pending),
            value: node.stats.value + lost * f64::from(node.pending),
            ..node.stats
        }
    }

    /// The result of the search so far.
    pub(super) fn result(&self) -> MctsResult<M> {
        let root = &self.nodes[0];
        let moves = (root.first..root.first + root.children).map(|child| {
            let node = &self.nodes[child];
            MoveStats {
                mv: node.mv.clone().expect("a child has a move"),
                visits: node.stats.visits,
                value: node.stats.value,
            }
        });
        let root: Vec<_> = moves.collect();

        MctsResult {
            best: self
                .most_visited(0)
                .and_then(|child| self.nodes[child].mv.clone()),
            root,
            pv: self.line(),
            stats: self.stats,
        }
    }

    /// The child of `index` with the most visits, ties going to the earliest.
    fn most_visited(&self, index: usize) -> Option<usize> {
        let node = &self.nodes[index];
        let visits = |child: usize| self.nodes[child].stats.visits;

        (node.first..node.first + node.children).reduce(|best, child| {
            if visits(child) > visits(best) {
                child
            } else {
                best
            }
        })
    }

    /// The most visited line from the root, while its nodes were visited.
    fn line(&self) -> Vec<M> {
        let mut line = Vec::new();
        let mut index = 0;
        while let Some(child) = self.most_visited(index) {
            if self.nodes[child].stats.visits == 0 {
                break;
            }

            line.extend(self.nodes[child].mv.clone());
            index = child;
        }

        line
    }
}

/// Makes `priors` a probability per move for `moves` moves: scaled to sum to one, or
/// uniform if they are missing, of another length, negative, not finite or all zero.
fn normalize(priors: &mut Vec<f64>, moves: usize) {
    let total: f64 = priors.iter().sum();
    let valid = priors.len() == moves
        && total.is_finite()
        && total > 0.0
        && priors
            .iter()
            .all(|&prior| prior.is_finite() && prior >= 0.0);
    if valid {
        for prior in priors.iter_mut() {
            *prior /= total;
        }
        return;
    }

    let uniform = 1.0 / f64::from(u32::try_from(moves.max(1)).unwrap_or(u32::MAX));
    priors.clear();
    priors.resize(moves, uniform);
}

/// Unit tests for prior normalization.
#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::normalize;

    /// Valid priors are scaled to sum to one; anything else becomes uniform.
    #[test]
    fn priors_become_probabilities() {
        let mut priors = vec![1.0, 3.0];
        normalize(&mut priors, 2);
        assert_eq!(priors, [0.25, 0.75]);

        for invalid in [
            vec![],
            vec![1.0],
            vec![0.0, 0.0],
            vec![-1.0, 2.0],
            vec![f64::NAN, 1.0],
        ] {
            let mut priors: Vec<f64> = invalid;
            normalize(&mut priors, 2);
            assert_eq!(priors, [0.5, 0.5]);
        }
    }
}

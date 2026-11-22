//! Alpha-beta pruning: minimax that skips moves which cannot change the decision.

use alloc::vec::Vec;

use crate::minimax::improves;
use crate::{Bound, Game, ScoredMove, SearchResult, Searcher, Side, Stats};

/// Searches `position` up to `depth` plies with fail-soft alpha-beta pruning.
///
/// Returns the [`minimax()`](crate::minimax()) score while visiting far fewer positions,
/// most so when [`Game::order_moves`] puts good moves first. Like minimax, the side to move
/// comes from [`Game::side_to_move`], so one side may move several times in a row.
///
/// With the default [`Game::satisfices`], the root score is the minimax value and its bound
/// is [`Bound::Exact`]; the best root move is exact too, while the other
/// [`SearchResult::root_moves`] are often only bounded. A game that overrides
/// [`Game::satisfices`] trades that exactness for speed, as documented there.
///
/// Allocates its working buffers on every call; [`Searcher::alpha_beta`] reuses them across
/// searches.
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
/// use game_tree_search::{Bound, alpha_beta, minimax};
///
/// let root = (10, Side::Max);
/// let pruned = alpha_beta(&Nim, &root, 10);
/// let full = minimax(&Nim, &root, 10);
/// assert_eq!((pruned.score, pruned.bound), (full.score, Bound::Exact));
/// assert!(pruned.stats.nodes < full.stats.nodes);
/// ```
#[must_use]
pub fn alpha_beta<G: Game>(
    game: &G,
    position: &G::Position,
    depth: u32,
) -> SearchResult<G::Move, G::Score> {
    Searcher::new().alpha_beta(game, position, depth)
}

impl<G: Game> Searcher<G> {
    /// Searches `position` up to `depth` plies with fail-soft alpha-beta pruning, reusing
    /// this searcher's buffers.
    ///
    /// Returns what [`alpha_beta()`](crate::alpha_beta()) returns. Once the buffers have grown
    /// to the search depth, a search allocates only the [`SearchResult`] it returns.
    #[must_use]
    pub fn alpha_beta(
        &mut self,
        game: &G,
        position: &G::Position,
        depth: u32,
    ) -> SearchResult<G::Move, G::Score> {
        self.stats = Stats::default();
        let mut root_moves = Vec::new();
        let node = self.search(
            game,
            position,
            0,
            depth,
            Window::FULL,
            Some(&mut root_moves),
        );

        SearchResult {
            score: node.score,
            bound: node.bound,
            pv: self.line(0).to_vec(),
            root_moves,
            unexamined: node.skipped,
            stats: self.stats,
        }
    }

    /// Searches `position`, `ply` moves below the root, with `depth` plies left within
    /// `window`, recording every child in `root_moves` (given only at the root).
    ///
    /// Leaves the position's principal variation in the line buffer of `ply`.
    fn search(
        &mut self,
        game: &G,
        position: &G::Position,
        ply: usize,
        depth: u32,
        mut window: Window<G::Score>,
        mut root_moves: Option<&mut Vec<ScoredMove<G::Move, G::Score>>>,
    ) -> Node<G::Score> {
        self.stats.nodes += 1;
        let mut moves = self.enter(ply);
        if depth > 0 {
            game.moves(position, &mut moves);
        }

        if moves.is_empty() {
            self.leave(ply, moves);
            return Node::leaf(game.evaluate(position));
        }

        game.order_moves(position, &mut moves);
        let side = game.side_to_move(position);
        let total = moves.len();
        let mut best: Option<Node<G::Score>> = None;
        let mut widened = false;
        let mut searched = 0;
        for mv in moves.drain(..) {
            if best.is_some_and(|best| stops(game, side, best.score, &window)) {
                break;
            }

            searched += 1;
            let child = game.play(position, &mv);
            let child = self.search(game, &child, ply + 1, depth - 1, window, None);
            let step = ScoredMove {
                mv,
                score: child.score,
                bound: child.bound,
            };
            if let Some(root_moves) = root_moves.as_deref_mut() {
                root_moves.push(step.clone());
            }

            widened |= child.bound.widens(side);
            window.tighten(side, child.score);
            if best.is_none_or(|best| improves(side, child.score, best.score)) {
                best = Some(child);
                self.adopt(ply, step);
            }
        }

        self.leave(ply, moves);
        let best = best.expect("a position with moves searches at least one");
        self.finish(side, best, total - searched, widened)
    }

    /// The position's result from its best child, the moves `skipped` and whether a searched
    /// child `widened` the value.
    fn finish(
        &mut self,
        side: Side,
        best: Node<G::Score>,
        skipped: usize,
        widened: bool,
    ) -> Node<G::Score> {
        let cut = skipped > 0;
        if cut {
            self.stats.cuts += 1;
        }

        Node {
            score: best.score,
            bound: Bound::combine(side, cut, best.bound, widened),
            skipped,
        }
    }
}

/// Whether `side`, holding `score`, skips its remaining moves: a window cutoff, or the game
/// is satisfied.
fn stops<G: Game>(game: &G, side: Side, score: G::Score, window: &Window<G::Score>) -> bool {
    window.closes(side, score) || game.satisfices(side, score)
}

/// The scores still worth telling apart at a position, from [`Side::Max`]'s point of view.
///
/// `None` is unbounded on that side.
#[derive(Debug, Clone, Copy)]
struct Window<S> {
    /// The score [`Side::Max`] is already sure of elsewhere.
    alpha: Option<S>,
    /// The score [`Side::Min`] is already sure of elsewhere.
    beta: Option<S>,
}

impl<S: Ord + Copy> Window<S> {
    /// The window that tells every score apart.
    const FULL: Self = Self {
        alpha: None,
        beta: None,
    };

    /// Whether `side`, holding `score`, can no longer affect the decision above.
    fn closes(&self, side: Side, score: S) -> bool {
        match side {
            Side::Max => self.beta.is_some_and(|beta| score >= beta),
            Side::Min => self.alpha.is_some_and(|alpha| score <= alpha),
        }
    }

    /// Narrows the window once `side` has found a move worth `score`.
    fn tighten(&mut self, side: Side, score: S) {
        match side {
            Side::Max => self.alpha = Some(self.alpha.map_or(score, |alpha| alpha.max(score))),
            Side::Min => self.beta = Some(self.beta.map_or(score, |beta| beta.min(score))),
        }
    }
}

/// What a search learned about one position; its principal variation sits in the
/// searcher's line buffer for its ply.
#[derive(Debug, Clone, Copy)]
struct Node<S> {
    /// The score, from [`Side::Max`]'s point of view.
    score: S,
    /// How [`Node::score`] relates to the position's minimax value.
    bound: Bound,
    /// The moves of the position left unsearched.
    skipped: usize,
}

impl<S> Node<S> {
    /// A position scored by [`Game::evaluate`]: the game ends or the depth runs out.
    const fn leaf(score: S) -> Self {
        Self {
            score,
            bound: Bound::Exact,
            skipped: 0,
        }
    }
}

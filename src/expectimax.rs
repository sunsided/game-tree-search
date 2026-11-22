//! Expectimax: minimax for games with chance positions, such as dice rolls or card draws.

use alloc::vec::Vec;

use crate::game::expanded_moves;
use crate::minimax::{improves, stand_pat};
use crate::result::{Line, RootMoves};
use crate::{Bound, Expectation, Game, ScoredMove, SearchResult, Stats};

/// A game with chance positions, where an outcome is drawn instead of a side choosing.
///
/// A chance position lists its outcomes, each a move with its probability; any other
/// position is played by a [`Side`](crate::Side) as in [`Game`]. Scores must be
/// [`Expectation`]s, so a chance position can be valued at the mean of its outcomes.
///
/// Only [`expectimax()`] reads the outcomes. Every other search sees a chance position
/// through [`Game::moves`] and [`Game::side_to_move`] alone: if `moves` lists nothing
/// there, as in the example below, [`alpha_beta()`](crate::alpha_beta()) scores it as a
/// finished game. Search games with chance positions with [`expectimax()`].
pub trait Stochastic: Game<Score: Expectation> {
    /// Appends the outcomes of `position` to `outcomes`, which arrives empty, if it is a
    /// chance position: each a move to [`Game::play`] and its probability.
    ///
    /// Probabilities are weights: they must be positive, but need not sum to one. Appending
    /// none makes the position an ordinary one, whose side to move chooses.
    fn outcomes(&self, position: &Self::Position, outcomes: &mut Vec<(Self::Move, f64)>);
}

/// Searches every line from `position` up to `depth` plies with expectimax.
///
/// Like [`minimax()`](crate::minimax()), except that a chance position is worth the
/// [`expectation`](Expectation::expectation) of its outcomes. An outcome counts as a ply, so
/// a chance position at the horizon is evaluated like any other. Quiescence works as in the
/// other searches, through [`Game::noisy_moves`]; a chance position is never quiesced.
///
/// The result is always [`Bound::Exact`]. At a chance position, the principal variation
/// follows the most likely outcome, so replaying it reaches a position whose evaluation is
/// that line's score, not the expectation; each step carries the expected value of the
/// position it leads to. At a chance root searched to depth 1 or more,
/// [`SearchResult::root_moves`] lists the outcomes with their values; at depth 0 the chance
/// root is only evaluated, so it lists none.
///
/// # Examples
///
/// Taking 1 point for sure, or gambling on 3 points against a loss of 2:
///
/// ```
/// use game_tree_search::{Game, Side, Stochastic, TotalF64, expectimax};
///
/// /// The choice, the gamble, and the end with its points.
/// #[derive(Debug, Clone, Copy, PartialEq)]
/// enum Bet {
///     Choose,
///     Gamble,
///     Done(f64),
/// }
///
/// /// A bet whose gamble wins with probability `odds`.
/// struct Wager {
///     odds: f64,
/// }
///
/// impl Game for Wager {
///     type Position = Bet;
///     type Move = Bet;
///     type Score = TotalF64;
///
///     fn side_to_move(&self, _: &Bet) -> Side {
///         Side::Max
///     }
///
///     fn moves(&self, position: &Bet, moves: &mut Vec<Bet>) {
///         if *position == Bet::Choose {
///             moves.extend([Bet::Done(1.0), Bet::Gamble]);
///         }
///     }
///
///     fn play(&self, _: &Bet, mv: &Bet) -> Bet {
///         *mv
///     }
///
///     fn evaluate(&self, position: &Bet) -> TotalF64 {
///         let points = if let Bet::Done(points) = *position { points } else { 0.0 };
///         TotalF64::new(points).expect("not NaN")
///     }
/// }
///
/// impl Stochastic for Wager {
///     fn outcomes(&self, position: &Bet, outcomes: &mut Vec<(Bet, f64)>) {
///         if *position == Bet::Gamble {
///             outcomes.extend([(Bet::Done(3.0), self.odds), (Bet::Done(-2.0), 1.0 - self.odds)]);
///         }
///     }
/// }
///
/// // Even odds: the gamble is worth 0.5, so the sure point wins.
/// let even = expectimax(&Wager { odds: 0.5 }, &Bet::Choose, 2);
/// assert_eq!((even.score.get(), even.best()), (1.0, Some(&Bet::Done(1.0))));
///
/// // At 75%, the gamble is worth 1.75.
/// let good = expectimax(&Wager { odds: 0.75 }, &Bet::Choose, 2);
/// assert_eq!((good.score.get(), good.best()), (1.75, Some(&Bet::Gamble)));
/// ```
#[must_use]
pub fn expectimax<G: Stochastic>(
    game: &G,
    position: &G::Position,
    depth: u32,
) -> SearchResult<G::Move, G::Score> {
    let mut stats = Stats::default();
    let mut root_moves = Vec::new();
    let (score, mut pv) = search(
        game,
        position,
        (0, depth),
        Some(&mut root_moves),
        &mut stats,
    );
    pv.reverse();

    SearchResult {
        score,
        bound: Bound::Exact,
        pv,
        root_moves,
        unexamined: 0,
        depth,
        proven: stats.horizon == 0,
        stats,
    }
}

/// The expectimax value of `position`, `ply` moves below the root with `depth` plies left,
/// and its principal variation in reverse (last move first).
///
/// Records every child of the root in `root_moves`, given only at the root.
fn search<G: Stochastic>(
    game: &G,
    position: &G::Position,
    (ply, depth): (u32, u32),
    root_moves: RootMoves<'_, G>,
    stats: &mut Stats,
) -> (G::Score, Line<G>) {
    let mut outcomes = Vec::new();
    game.outcomes(position, &mut outcomes);
    if outcomes.is_empty() {
        return choose(game, position, (ply, depth), root_moves, stats);
    }

    stats.visit(ply);
    if depth == 0 {
        stats.horizon += 1;
        return (game.evaluate(position), Vec::new());
    }

    draw(game, position, (ply, depth), (outcomes, root_moves), stats)
}

/// The value of the chance `position`: the expectation over its `outcomes`, with the line of
/// the most likely one.
fn draw<G: Stochastic>(
    game: &G,
    position: &G::Position,
    (ply, depth): (u32, u32),
    (outcomes, mut root_moves): (Vec<(G::Move, f64)>, RootMoves<'_, G>),
    stats: &mut Stats,
) -> (G::Score, Line<G>) {
    let mut values = Vec::with_capacity(outcomes.len());
    let mut likeliest: Option<(f64, Line<G>)> = None;
    for (mv, weight) in outcomes {
        let child = game.play(position, &mv);
        let (score, mut line) = search(game, &child, (ply + 1, depth - 1), None, stats);
        values.push((score, weight));
        if let Some(root_moves) = root_moves.as_deref_mut() {
            root_moves.push(ScoredMove::exact(mv.clone(), score));
        }

        if likeliest.as_ref().is_none_or(|&(best, _)| weight > best) {
            line.push(ScoredMove::exact(mv, score));
            likeliest = Some((weight, line));
        }
    }

    let (_, line) = likeliest.expect("a chance position has outcomes");
    (G::Score::expectation(&values), line)
}

/// The value of `position`, where a side chooses, as in minimax.
fn choose<G: Stochastic>(
    game: &G,
    position: &G::Position,
    (ply, depth): (u32, u32),
    mut root_moves: RootMoves<'_, G>,
    stats: &mut Stats,
) -> (G::Score, Line<G>) {
    stats.visit(ply);
    let moves = expanded_moves(game, position, depth, stats);
    if moves.is_empty() {
        return (game.evaluate(position), Vec::new());
    }

    let side = game.side_to_move(position);
    let mut best = stand_pat(game, position, depth);
    for mv in moves {
        let child = game.play(position, &mv);
        let next = (ply + 1, depth.saturating_sub(1));
        let (score, mut line) = search(game, &child, next, None, stats);
        if let Some(root_moves) = root_moves.as_deref_mut() {
            root_moves.push(ScoredMove::exact(mv.clone(), score));
        }

        if best
            .as_ref()
            .is_none_or(|&(best, _)| improves(side, score, best))
        {
            line.push(ScoredMove::exact(mv, score));
            best = Some((score, line));
        }
    }

    best.expect("a position with moves has a best child")
}

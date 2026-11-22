//! How exact a search score is: [`Bound`], and how a position's bound follows from its
//! children's.

use crate::{Score, Side};

/// How a reported score relates to the true minimax value.
///
/// Pruning stops a search below a position once the rest cannot change the decision above
/// it, or once the side to move is satisfied (see [`Game::satisfices`](crate::Game::satisfices)).
/// The score reached by then can be a bound instead of the value itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Bound {
    /// The score is the minimax value.
    Exact,
    /// The minimax value is at least the score.
    Lower,
    /// The minimax value is at most the score.
    Upper,
    /// The minimax value may lie on either side of the score.
    ///
    /// Only arises where [`Game::satisfices`](crate::Game::satisfices) stopped a search.
    Approximate,
}

impl Bound {
    /// Whether the minimax value of a score with this bound may lie above the score.
    #[must_use]
    pub const fn open_above(self) -> bool {
        matches!(self, Self::Lower | Self::Approximate)
    }

    /// Whether the minimax value of a score with this bound may lie below the score.
    #[must_use]
    pub const fn open_below(self) -> bool {
        matches!(self, Self::Upper | Self::Approximate)
    }

    /// Whether a minimax value of `value` is consistent with `score` under this bound.
    ///
    /// # Examples
    ///
    /// ```
    /// use game_tree_search::Bound;
    ///
    /// assert!(Bound::Lower.admits(3, 5));
    /// assert!(!Bound::Upper.admits(3, 5));
    /// assert!(Bound::Exact.admits(3, 3));
    /// ```
    #[must_use]
    pub fn admits<S: Score>(self, score: S, value: S) -> bool {
        (value <= score || self.open_above()) && (value >= score || self.open_below())
    }

    /// The bound of a position's value, from what is known about its children.
    ///
    /// `side` moves in the position. `cut` tells whether it stopped with moves left unsearched.
    /// `best` is the bound of the child that set the position's score. `widened` tells
    /// whether any searched child may be better for `side` than its score says.
    ///
    /// The best child pins the side `side` optimizes away from; the other side is only
    /// closed if no child, searched or not, can beat the score.
    pub(crate) const fn combine(side: Side, cut: bool, best: Self, widened: bool) -> Self {
        let complete = !cut && !widened;
        let (below_closed, above_closed) = match side {
            Side::Max => (!best.open_below(), complete),
            Side::Min => (complete, !best.open_above()),
        };

        match (below_closed, above_closed) {
            (true, true) => Self::Exact,
            (true, false) => Self::Lower,
            (false, true) => Self::Upper,
            (false, false) => Self::Approximate,
        }
    }

    /// Whether a child with this bound may be better for `side` than its score says.
    pub(crate) const fn widens(self, side: Side) -> bool {
        match side {
            Side::Max => self.open_above(),
            Side::Min => self.open_below(),
        }
    }
}

/// Unit tests for bound combination.
#[cfg(test)]
mod tests {
    use super::Bound::{self, Approximate, Exact, Lower, Upper};
    use crate::Side::{self, Max, Min};

    /// Every combination of cut, best bound and widening yields the expected bound.
    #[test]
    fn combine_covers_all_outcomes() {
        let cases: [(Side, bool, Bound, bool, Bound); 18] = [
            (Max, false, Exact, false, Exact),
            (Max, false, Lower, false, Exact),
            (Max, false, Upper, false, Upper),
            (Max, false, Approximate, false, Upper),
            (Max, true, Exact, false, Lower),
            (Max, false, Exact, true, Lower),
            (Max, true, Lower, true, Lower),
            (Max, true, Upper, false, Approximate),
            (Max, false, Approximate, true, Approximate),
            (Min, false, Exact, false, Exact),
            (Min, false, Upper, false, Exact),
            (Min, false, Lower, false, Lower),
            (Min, false, Approximate, false, Lower),
            (Min, true, Exact, false, Upper),
            (Min, false, Exact, true, Upper),
            (Min, true, Upper, true, Upper),
            (Min, true, Lower, false, Approximate),
            (Min, false, Approximate, true, Approximate),
        ];
        for (side, cut, best, widened, expected) in cases {
            assert_eq!(
                Bound::combine(side, cut, best, widened),
                expected,
                "{side:?} cut={cut} best={best:?} widened={widened}"
            );
        }
    }

    /// A bound admits exactly the values on its open sides.
    #[test]
    fn admits_follows_the_open_sides() {
        assert!(Exact.admits(2, 2));
        assert!(!Exact.admits(2, 3));
        assert!(Upper.admits(2, 1));
        assert!(!Lower.admits(2, 1));
        assert!(Approximate.admits(2, 1) && Approximate.admits(2, 3));
    }
}

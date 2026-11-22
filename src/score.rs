//! Scores of positions: the [`Score`] trait and the totally ordered floats [`TotalF32`] and
//! [`TotalF64`].

use core::cmp::Ordering;

/// A position's value, from the point of view of [`Side::Max`](crate::Side::Max).
///
/// Higher is better for [`Side::Max`](crate::Side::Max), lower is better for
/// [`Side::Min`](crate::Side::Min). Scores need a total order, so raw floats (where NaN
/// compares with nothing) are wrapped in [`TotalF32`] or [`TotalF64`].
///
/// Implemented for the signed integers and for [`TotalF32`] and [`TotalF64`]. A game whose
/// outcomes carry more than a number, such as a win, a loss or a retreat, can implement it
/// for its own type, as long as the order ranks every outcome by how good it is for
/// [`Side::Max`](crate::Side::Max).
pub trait Score: Copy + Ord {
    /// The same score from the other side's point of view.
    ///
    /// Must be an order-reversing involution: `a < b` implies `b.negate() < a.negate()`,
    /// and `s.negate().negate() == s` for every score. Negamax relies on both to return
    /// exactly the minimax score.
    ///
    /// This need not be arithmetic negation. The integer implementations use bitwise NOT
    /// (`-s - 1`), which mirrors the whole range, `MIN` and `MAX` included, where
    /// arithmetic negation would overflow at `MIN`.
    #[must_use]
    fn negate(self) -> Self;

    /// A score strictly above `self`, or `self` if no score is above it.
    ///
    /// Null-window searches such as [`pvs()`](crate::pvs()) use it to ask "is the value above
    /// this score?". Any strictly greater score is correct; the next representable one asks
    /// the sharpest question. The integer implementations add one, the floats step to the
    /// next value in [`f32::total_cmp`] order.
    #[must_use]
    fn next_up(self) -> Self;

    /// A score strictly below `self`, or `self` if no score is below it; the mirror of
    /// [`Score::next_up`].
    #[must_use]
    fn next_down(self) -> Self;
}

/// Implements [`Score`] for signed integer types, negating by bitwise NOT.
macro_rules! integer_score {
    ($($ty:ty),*) => {
        $(
            impl Score for $ty {
                fn negate(self) -> Self {
                    !self
                }

                fn next_up(self) -> Self {
                    self.saturating_add(1)
                }

                fn next_down(self) -> Self {
                    self.saturating_sub(1)
                }
            }
        )*
    };
}

integer_score!(i8, i16, i32, i64, i128, isize);

/// Defines a float wrapper with a total order that rejects NaN.
macro_rules! total_float {
    ($(#[$doc:meta])* $name:ident, $float:ty, $bits:ty, $narrow:path) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name($float);

        impl $name {
            /// Wraps `value`, or returns `None` if it is NaN.
            #[must_use]
            pub fn new(value: $float) -> Option<Self> {
                (!value.is_nan()).then_some(Self(value))
            }

            /// The wrapped value; never NaN.
            #[must_use]
            pub const fn get(self) -> $float {
                self.0
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.cmp(other) == Ordering::Equal
            }
        }

        impl Eq for $name {}

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> Ordering {
                self.0.total_cmp(&other.0)
            }
        }

        impl Score for $name {
            fn negate(self) -> Self {
                Self(-self.0)
            }

            fn next_up(self) -> Self {
                let bits = self.0.to_bits();
                let up = if self.0 == <$float>::INFINITY {
                    bits
                } else if self.0.is_sign_positive() {
                    bits + 1
                } else if self.0 == 0.0 {
                    0
                } else {
                    bits - 1
                };

                Self(<$float>::from_bits(up))
            }

            fn next_down(self) -> Self {
                self.negate().next_up().negate()
            }
        }

        impl Expectation for $name {
            /// The weighted mean, with each weight first divided by the largest so neither
            /// the weights nor the products overflow; infinite scores of both signs average
            /// to NaN, which is returned as zero.
            fn expectation(outcomes: &[(Self, f64)]) -> Self {
                let largest = outcomes.iter().map(|&(_, weight)| weight).fold(0.0, f64::max);
                let total: f64 = outcomes.iter().map(|&(_, weight)| weight / largest).sum();
                let mean: f64 = outcomes
                    .iter()
                    .map(|&(score, weight)| f64::from(score.0) * (weight / largest / total))
                    .sum();

                Self::new($narrow(mean)).unwrap_or_default()
            }
        }
    };
}

/// A [`Score`] that can be averaged, as chance positions need.
///
/// Implemented for [`TotalF32`] and [`TotalF64`]; [`expectimax()`](crate::expectimax())
/// requires it.
pub trait Expectation: Score {
    /// The mean of `outcomes`, each a score with its probability.
    ///
    /// Probabilities are weights: they need not sum to one, only be positive. Never called
    /// with no outcomes.
    #[must_use]
    fn expectation(outcomes: &[(Self, f64)]) -> Self;
}

total_float!(
    /// An `f32` score that is never NaN, ordered by [`f32::total_cmp`].
    ///
    /// The order puts `-0.0` just below `0.0` and the infinities at the ends.
    ///
    /// # Examples
    ///
    /// ```
    /// use game_tree_search::{Score, TotalF32};
    ///
    /// let win = TotalF32::new(5.0).expect("not NaN");
    /// assert!(win.negate() < win);
    /// assert!(TotalF32::new(f32::NAN).is_none());
    /// ```
    TotalF32,
    f32,
    u32,
    narrow_to_f32
);

total_float!(
    /// An `f64` score that is never NaN, ordered by [`f64::total_cmp`].
    ///
    /// The order puts `-0.0` just below `0.0` and the infinities at the ends.
    TotalF64,
    f64,
    u64,
    core::convert::identity
);

/// `value` rounded to the nearest `f32`, for the mean of `f32` scores.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a mean of f32 scores fits an f32 again"
)]
const fn narrow_to_f32(value: f64) -> f32 {
    value as f32
}

/// Unit tests for the integer and float scores.
#[cfg(test)]
mod tests {
    use super::{Score, TotalF32, TotalF64};

    /// Integer negation is an order-reversing involution over the whole range.
    #[test]
    fn integer_negation_mirrors_the_whole_range() {
        assert_eq!(3_i32.negate(), -4);
        assert_eq!(i8::MIN.negate(), i8::MAX);
        assert_eq!(i8::MAX.negate(), i8::MIN);
        for s in i8::MIN..i8::MAX {
            assert!((s + 1).negate() < s.negate(), "{s}");
            assert_eq!(s.negate().negate(), s, "{s}");
        }
    }

    /// Floats order totally, with the infinities at the ends.
    #[test]
    fn floats_order_totally() {
        let f = |value| TotalF64::new(value).expect("not NaN");
        assert!(f(f64::NEG_INFINITY) < f(-1.0));
        assert!(f(-0.0) < f(0.0));
        assert!(f(1.0) < f(f64::INFINITY));
        assert_eq!(f(2.0).negate(), f(-2.0));
    }

    /// Integer steps add or remove one and stop at the ends of the range.
    #[test]
    fn integer_steps_saturate() {
        assert_eq!((3_i32.next_up(), 3_i32.next_down()), (4, 2));
        assert_eq!((i8::MAX.next_up(), i8::MIN.next_down()), (i8::MAX, i8::MIN));
    }

    /// Float steps walk the total order: through both zeros, the subnormals and up to the
    /// infinities, where they stop.
    #[test]
    fn float_steps_walk_the_total_order() {
        let f = |value| TotalF32::new(value).expect("not NaN");
        let tiny = f32::from_bits(1);
        assert_eq!(f(-0.0).next_up().get().to_bits(), 0.0_f32.to_bits());
        assert_eq!(f(0.0).next_up().get(), tiny);
        assert_eq!(f(0.0).next_down().get().to_bits(), (-0.0_f32).to_bits());
        assert_eq!(f(-0.0).next_down().get(), -tiny);
        assert_eq!(f(f32::MAX).next_up().get(), f32::INFINITY);
        assert_eq!(f(f32::INFINITY).next_up().get(), f32::INFINITY);
        assert_eq!(f(f32::NEG_INFINITY).next_down().get(), f32::NEG_INFINITY);
        assert_eq!(f(f32::NEG_INFINITY).next_up().get(), -f32::MAX);
        assert!(f(1.0).next_up() > f(1.0) && f(1.0).next_down() < f(1.0));
        assert!(f(-1.0).next_up() > f(-1.0));
        assert_eq!(
            TotalF64::new(0.0).map(|zero| zero.next_up().get()),
            Some(f64::from_bits(1))
        );
    }

    /// The expectation weighs scores by their probabilities, which need not sum to one, and
    /// turns the NaN of opposite infinities into zero.
    #[test]
    fn expectations_are_weighted_means() {
        use super::Expectation;

        let f = |value| TotalF64::new(value).expect("not NaN");
        assert_eq!(
            TotalF64::expectation(&[(f(3.0), 1.0), (f(-1.0), 3.0)]).get(),
            0.0
        );
        assert_eq!(TotalF64::expectation(&[(f(2.0), 0.5)]).get(), 2.0);
        let opposite = [(f(f64::INFINITY), 1.0), (f(f64::NEG_INFINITY), 1.0)];
        assert_eq!(TotalF64::expectation(&opposite).get(), 0.0);
        let huge = [(f(0.001), 1e308), (f(0.002), 1e308)];
        assert!((TotalF64::expectation(&huge).get() - 0.0015).abs() < 1e-15);
        assert_eq!(TotalF64::expectation(&[(f(f64::MAX), 2.0)]).get(), f64::MAX);
        let single = TotalF32::new(1.5).expect("not NaN");
        assert_eq!(TotalF32::expectation(&[(single, 2.0)]).get(), 1.5);
    }

    /// NaN is rejected, so the order never meets it.
    #[test]
    fn nan_is_rejected() {
        assert!(TotalF32::new(f32::NAN).is_none());
        assert_eq!(TotalF32::new(1.5).map(TotalF32::get), Some(1.5));
    }
}

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
    /// Must reverse the order: `a < b` implies `b.negate() <= a.negate()`. The integer
    /// implementations saturate, so `MIN.negate()` is `MAX`; keep integer scores above
    /// `MIN` to make the order strictly reversed.
    #[must_use]
    fn negate(self) -> Self;
}

/// Implements [`Score`] for signed integer types with saturating negation.
macro_rules! integer_score {
    ($($ty:ty),*) => {
        $(
            impl Score for $ty {
                fn negate(self) -> Self {
                    self.saturating_neg()
                }
            }
        )*
    };
}

integer_score!(i8, i16, i32, i64, i128, isize);

/// Defines a float wrapper with a total order that rejects NaN.
macro_rules! total_float {
    ($(#[$doc:meta])* $name:ident, $float:ty) => {
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
        }
    };
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
    f32
);

total_float!(
    /// An `f64` score that is never NaN, ordered by [`f64::total_cmp`].
    ///
    /// The order puts `-0.0` just below `0.0` and the infinities at the ends.
    TotalF64,
    f64
);

/// Unit tests for the integer and float scores.
#[cfg(test)]
mod tests {
    use super::{Score, TotalF32, TotalF64};

    /// Integer negation reverses the order and saturates at the minimum.
    #[test]
    fn integer_negation_reverses_and_saturates() {
        assert_eq!(3_i32.negate(), -3);
        assert!(1_i32.negate() < 0_i32.negate());
        assert_eq!(i8::MIN.negate(), i8::MAX);
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

    /// NaN is rejected, so the order never meets it.
    #[test]
    fn nan_is_rejected() {
        assert!(TotalF32::new(f32::NAN).is_none());
        assert_eq!(TotalF32::new(1.5).map(TotalF32::get), Some(1.5));
    }
}

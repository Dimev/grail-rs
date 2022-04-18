/// the number of formants to synthesize
/// TODO: consider using 8 instead?
pub const NUM_FORMANTS: usize = 12;

// we'll want to implement these for arrays
use core::ops::{Add, Div, Mul, Sub};

// and some arithmatic functions
// these are approximations to help speed things up
// hyperbolic tangent, x is multiplied by pi
/// Approximation of the hyperbolic tangent, tan(pi*x).
/// Approximation is good for x = [0.0; 0.5]
#[inline]
pub fn tan_approx(x: f32) -> f32 {
    // tan(x) = sin(x) / cos(x)
    // we can approximate sin and x with the bhaskara I approximation quite well
    // which is 16x(pi - x) / 5pi^2 - 4x(pi - x) for sin
    // if we fill it in, multiply pi by and rewrite it, we get this:
    ((1.0 - x) * x * (5.0 - 4.0 * (x + 0.5) * (0.5 - x)))
        / ((x + 0.5) * (5.0 - 4.0 * (1.0 - x) * x) * (0.5 - x))
}

// next, let's make a struct to help storing arrays, and do operations on them

/// Array, containing NUM_FORMANTS floats. Used to store per-formant data
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct Array {
    /// inner array
    pub x: [f32; NUM_FORMANTS],
}

impl Array {
    /// makes a new Array from a given array
    #[inline]
    pub fn new(arr: [f32; NUM_FORMANTS]) -> Self {
        Self { x: arr }
    }

	/// make a new array from a given function
	#[inline]
	pub fn from_func<F: FnMut() -> f32>(f: &mut F) -> Self {
		let mut arr = [0.0; NUM_FORMANTS];
		for x in arr.iter_mut() {
			*x = f();
		}
		Self { x: arr }
	}

    /// makes a new array and fills it with a single element
    #[inline]
    pub fn splat(val: f32) -> Self {
        Self {
            x: [val; NUM_FORMANTS],
        }
    }

	// TODO: use this for everything
	/// do something for every value in the array
	#[inline]
	pub fn map<F: Fn(f32) -> f32>(mut self, f: F) -> Self {		
        for i in 0..NUM_FORMANTS {
            self.x[i] = f(self.x[i]);
        }
        self
	}

    /// sums all elements in an array together
    #[inline]
    pub fn sum(self) -> f32 {
        let mut res = 0.0;
        for i in 0..NUM_FORMANTS {
            res += self.x[i];
        }
        res
    }

    /// blend two arrays, based on some blend value
    #[inline]
    pub fn blend(self, other: Self, alpha: f32) -> Self {
        let mut res = self;
        for i in 0..NUM_FORMANTS {
            res.x[i] *= 1.0 - alpha;
            res.x[i] += other.x[i] * alpha;
        }
        res
    }

    /// blend two arrays, based on some blend array
    #[inline]
    pub fn blend_multiple(self, other: Self, alpha: Array) -> Self {
        self * (Array::splat(1.0) - alpha) + other * alpha
    }

    /// hyperbolic tangent approximation
    #[inline]
    pub fn tan_approx(self) -> Self {
        let mut res = self;
        for i in 0..NUM_FORMANTS {
            res.x[i] = tan_approx(res.x[i])
        }
        res
    }

    /// cos function
    #[inline]
    pub fn cos(self) -> Self {
        let mut res = self;
        for i in 0..NUM_FORMANTS {
            res.x[i] = res.x[i].cos();
        }
        res
    }

    /// exp function
    #[inline]
    pub fn exp(self) -> Self {
        let mut res = self;
        for i in 0..NUM_FORMANTS {
            res.x[i] = res.x[i].exp();
        }
        res
    }

    /// fract, take away the integer part of the number
    #[inline]
    pub fn fract(self) -> Self {
        let mut res = self;
        for i in 0..NUM_FORMANTS {
            res.x[i] = res.x[i].fract();
        }
        res
    }

    /// floor, leave only the integer part of the number
    #[inline]
    pub fn floor(self) -> Self {
        let mut res = self;
        for i in 0..NUM_FORMANTS {
            res.x[i] = res.x[i].floor();
        }
        res
    }
}

// and arithmatic
// using the Op  traits to make life easier here, this way we can just do +, - * and /
impl Add for Array {
    type Output = Self;
    /// adds two arrays together
    #[inline]
    fn add(self, other: Self) -> Self {
        let mut res = self;
        for i in 0..NUM_FORMANTS {
            res.x[i] += other.x[i];
        }
        res
    }
}

impl Sub for Array {
    type Output = Self;
    /// subtracts an array from another
    #[inline]
    fn sub(self, other: Self) -> Self {
        let mut res = self;
        for i in 0..NUM_FORMANTS {
            res.x[i] -= other.x[i];
        }
        res
    }
}

impl Mul for Array {
    type Output = Self;
    /// multiplies two arrays together
    #[inline]
    fn mul(self, other: Self) -> Self {
        let mut res = self;
        for i in 0..NUM_FORMANTS {
            res.x[i] *= other.x[i];
        }
        res
    }
}

impl Div for Array {
    type Output = Self;
    /// divides one array with another
    #[inline]
    fn div(self, other: Self) -> Self {
        let mut res = self;
        for i in 0..NUM_FORMANTS {
            res.x[i] /= other.x[i];
        }
        res
    }
}

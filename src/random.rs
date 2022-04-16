use crate::array::*;

/// generates a random float in the range [-1, 1], and changes the state after doing so
#[inline]
pub fn random_f32(state: &mut u32) -> f32 {
    // here we change the state with a regular integer rng
    // This is the lehmer random number generator: https://en.wikipedia.org/wiki/Lehmer_random_number_generator
    // 16807 here is a magic number. In theory this could be any coprime, but there are some numbers that work better
    *state = state.wrapping_mul(16807).wrapping_add(1);

    // https://experilous.com/1/blog/post/perfect-fast-random-floating-point-numbers
    // and here we get the right part of the integer to generate our float from
    // this abuses IEE 754 floats (and works with doubles too)
    // the first 9 bits of the float are the sign bit, and the exponent
    // numbers from 1 - 2 in this have the same exponent (which the | 0x3F800000 sets)
    // then we can set the mantissa with the state
    // we shift that to the right so the first 9 bits become 0, and don't affect our exponent
    // for doubles (f64) we need to shift by 12, due to the sign and exponent taking up 12 bits, and set these to 0x3FF0000000000000 instead
    let res = (*state >> 9) | 0x3F800000;

    // and here we get the float number
    // we have a range of 1-2, but we want -1 to 1
    (f32::from_bits(res) - 1.5) * 2.0
}

// first, we want to make a few structs to help with generating noise
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct ValueNoise {
    current: f32,
    next: f32,
    phase: f32,
    state: u32,
}

impl ValueNoise {
    pub fn new(state: &mut u32) -> Self {
        let current = random_f32(state);
        let next = random_f32(state);

        Self {
            current,
            next,
            phase: 0.0,
            state: *state,
        }
    }

    pub fn next(&mut self, increment: f32) -> f32 {
        // increment the state
        self.phase += increment;

        // wrap it around if needed
        if self.phase > 1.0 {
            self.phase -= 1.0;

            // also update the noise
            self.current = self.next;
            self.next = random_f32(&mut self.state);
        }

        // and blend between the current and next
        self.current * (1.0 - self.phase) + self.next * self.phase
    }
}

// and for arrays too
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct ArrayValueNoise {
    current: Array,
    next: Array,
    phase: f32,
    state: u32,
}

impl ArrayValueNoise {
    pub fn new(state: &mut u32) -> Self {
        let mut current = [0.0; NUM_FORMANTS];
        let mut next = [0.0; NUM_FORMANTS];

        // write to the arrays
        for i in 0..NUM_FORMANTS {
            current[i] = random_f32(state);
            next[i] = random_f32(state);
        }

        Self {
            current: Array::new(current),
            next: Array::new(next),
            phase: 0.0,
            state: *state,
        }
    }

    pub fn next(&mut self, increment: f32) -> Array {
        // increment the state
        self.phase += increment;

        // wrap it around if needed
        if self.phase > 1.0 {
            self.phase -= 1.0;

            // also update the noise
            self.current = self.next;

            for i in 0..NUM_FORMANTS {
                self.next.x[i] = random_f32(&mut self.state);
            }
        }

        // and blend between the current and next
        self.current * Array::splat(1.0 - self.phase) + self.next * Array::splat(self.phase)
    }
}

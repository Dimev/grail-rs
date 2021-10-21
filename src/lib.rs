#![no_std]
#![forbid(unsafe_code)]

// The main file the synth is in
// first, define some constants

/// default sample rate all voices use
/// Resampling to a different sample rate is possible
pub const DEFAULT_SAMPLE_RATE: u32 = 44100;

/// the number of formants to synthesize
pub const NUM_FORMANTS: usize = 12;

/// the number of formants that are voiced, and don't receive noise as input
pub const NUM_VOICED_FORMANTS: usize = 8;

// next, let's make a struct to help storing arrays, and do operations on them

/// Array, containing NUM_FORMANTS elements. Used to store formants
#[derive(Copy, Clone, Debug)]
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

    /// makes a new array and fills it with a single element
    #[inline]
    pub fn splat(val: f32) -> Self {
        Self {
            x: [val; NUM_FORMANTS],
        }
    }

    // and arithmatic
    // not using the OP traits here to keep it simple

    /// adds two arrays together
    #[inline]
    pub fn add(self, other: Self) -> Self {
        let mut res = self;
        for i in 0..NUM_FORMANTS {
            res.x[i] += other.x[i];
        }
        res
    }

    /// subtracts an array from another
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        let mut res = self;
        for i in 0..NUM_FORMANTS {
            res.x[i] -= other.x[i];
        }
        res
    }

    /// multiplies two arrays together
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        let mut res = self;
        for i in 0..NUM_FORMANTS {
            res.x[i] *= other.x[i];
        }
        res
    }

    /// divides one array with another
    #[inline]
    pub fn div(self, other: Self) -> Self {
        let mut res = self;
        for i in 0..NUM_FORMANTS {
            res.x[i] /= other.x[i];
        }
        res
    }

    /// sums all elements in an array together
    #[inline]
    pub fn sum(self) -> f32 {
        self.x.iter().sum()
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
}

// and, a helper function to do random number generation

/// generates a random float, and changes the state after doing so
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
    return (f32::from_bits(res) - 1.5) * 2.0;
}

// next up, let's go to the audio part
// we'll want a way to represent what to synthesize

/// synthesis element, describes what to synthesize
#[derive(Copy, Clone, Debug)]
pub struct SynthesisElem {
    /// base frequency, normalized to sample rate
    pub frequency: f32,

    /// formant frequencies, normalized to sample rate
    pub formant_freq: Array,

    /// formant bandwidths, normalized to sample rate
    pub formant_bw: Array,

    /// formant amplitudes. If these sum up to one, the output amplitude will also be one
    pub formant_amp: Array,

    /// antiresonator frequency, normalized to sample rate
    pub nasal_freq: f32,

    /// antiresonator bandwidth
    pub nasal_bw: f32,

    /// antiresonator amplitude
    pub nasal_amp: f32,

    /// voice softness, 0 is saw, 1 is sine
    pub softness: f32,
}

// next, make some functions for the element
// we want to make one from some sample rate, make one with the given sample rate, and blend them
impl SynthesisElem {
    // make a new one

    // make a new one with the default sample rate, and unit gain

    /// blend between this synthesis element and another one
    #[inline]
    pub fn blend(self, other: Self, alpha: f32) -> Self {
        Self {
            frequency: self.frequency * (1.0 - alpha) + other.frequency * alpha,
            formant_freq: self.formant_freq.blend(other.formant_freq, alpha),
            formant_bw: self.formant_bw.blend(other.formant_bw, alpha),
            formant_amp: self.formant_amp.blend(other.formant_amp, alpha),
            nasal_freq: self.nasal_freq * (1.0 - alpha) + other.nasal_freq * alpha,
            nasal_bw: self.nasal_bw * (1.0 - alpha) + other.nasal_bw * alpha,
            nasal_amp: self.nasal_amp * (1.0 - alpha) + other.nasal_amp * alpha,
            softness: self.softness * (1.0 - alpha) + other.softness * alpha,
        }
    }

    /// resample the synthesis element to a new sample rate
    #[inline]
    pub fn resample(self, old_sample_rate: u32, new_sample_rate: u32) -> Self {
        // scale factor for the sample rate
        let scale = old_sample_rate as f32 / new_sample_rate as f32;

        Self {
            frequency: self.frequency * scale,
            formant_freq: self.formant_freq.mul(Array::splat(scale)),
            nasal_freq: self.nasal_freq * scale,
            ..self // this means fill in the rest of the struct with self
        }
    }
}

// next we'll want to synthesize some audio
// for that, we'll use an iterator
// it keeps track of the filter states, and the underlying iterator to get synthesis elements from
// if iterators aren't available in the language you are porting to, use a function to get the next item from some state instead
pub struct Synthesize<T: Iterator<Item = SynthesisElem>> {
    /// underlying iterator
    iter: T,

    /// filter state a
    formant_state_a: Array,

    /// filter state b
    formant_state_b: Array,

    /// antiresonator state a
    nasal_state_a: f32,

    /// antiresonator state b
    nasal_state_b: f32,

    /// phase for the current pulse
    phase: f32,

    /// noise state
    seed: u32,
}

// TODO: voice here?
// needed because we probably want jitter to read it's parameters from voice, but we can do that later if really needed, and just pass voice.param_a in there

// next up, implement iterator for the synthesizer, which takes care of synthesizing sound (in samples) from synthesis elements
impl<T: Iterator<Item = SynthesisElem>> Iterator for Synthesize<T> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        // get the item from the underlying iterator, or return None if we can't
        let elem = self.iter.next()?;

        // first, we want to generate the impulse to put through the filter
        // this is a blend between a saw wave, and a triangle wave, then passed through a smoothstep
        let rising_phase = self.phase / (1.0 - 0.5 * elem.softness);
        let falling_phase = (1.0 - self.phase) / (0.5 * elem.softness);

        // make the triangle/saw wave
        let wave = rising_phase.min(falling_phase);

        // and pass it through the smoothstep
        let pulse = 6.0 * wave * wave - 4.0 * wave * wave * wave - 1.0;

        // increment the phase
        self.phase += elem.frequency;

        // and wrap it back around if needed
        if self.phase > 1.0 {
            self.phase -= 1.0;
        }

        // and also generate the noise
        let noise = random_f32(&mut self.seed);

        // now put these in an array, this way avoids using mut directly
        let x = Array::new({
            // fill it with noise
            let mut arr = [noise; NUM_FORMANTS];

            // and set the first n to the pulse
            for i in 0..NUM_VOICED_FORMANTS {
                arr[i] = pulse;
            }

            // and return the array to put it in x
            arr
        });

        // now, we can apply the first filter, using the array arithmatic

        // now, sum up all the filters, as they were (hopefully) done in parallel

        // and now, do the antiresonator on the summed value

        // and return the found value
        Some(x.sum())
    }
}

// and we want to be able to easily make a synthesizer, so make a trait for it
pub trait IntoSynthesize
where
    Self: IntoIterator<Item = SynthesisElem> + Sized,
{
    /// creates a new synthesizer from this iterator
    fn synthesize(self) -> Synthesize<Self::IntoIter> {
        Synthesize {
            iter: self.into_iter(),
            formant_state_a: Array::splat(0.0),
            formant_state_b: Array::splat(0.0),
            nasal_state_a: 0.0,
            nasal_state_b: 0.0,
            phase: 0.0,
            seed: 0,
        }
    }
}

// implement it for anything that can become the right iterator
impl<T> IntoSynthesize for T where T: IntoIterator<Item = SynthesisElem> + Sized {}

// that's it, sound synthesis done
// We also want to jitter all frequencies a bit for more realism, so let's do that next

// first, we want to make a few structs to help with generating noise
#[derive(Copy, Clone, Debug)]
struct ValueNoise {
    current: f32,
    next: f32,
    phase: f32,
    state: u32,
}

impl ValueNoise {
    fn new(state: &mut u32) -> Self {
        let current = random_f32(state);
        let next = random_f32(state);

        Self {
            current,
            next,
            phase: 0.0,
            state: *state,
        }
    }

    fn next(&mut self, increment: f32) -> f32 {
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
#[derive(Copy, Clone, Debug)]
struct ArrayValueNoise {
    current: Array,
    next: Array,
    phase: f32,
    state: u32,
}

impl ArrayValueNoise {
    fn new(state: &mut u32) -> Self {
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

    fn next(&mut self, increment: f32) -> Array {
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
        self.current
            .mul(Array::splat(1.0 - self.phase))
            .add(self.next.mul(Array::splat(self.phase)))
    }
}

// now we can make our jitter work, as getting random numbers is now easier
pub struct Jitter<T: Iterator<Item = SynthesisElem>> {
    /// underlying iterator
    iter: T,

    /// noise for the frequency
    freq_noise: ValueNoise,

    /// noise for the formant frequency
    formant_noise: ArrayValueNoise,

    /// noise for the formant amplitude
    amplitude_noise: ArrayValueNoise,
    // nasal frequency

    // nasal amplitude

    // noise frequency

    // frequency deviation

    // formant deviation

    // amplitude deviation
}

// Here's how it will work
// synthesizer iterator to generate sound
// jitter iterator to add randomness to the frequencies
// sequencer iterator to blend phonemes
// intonator to add intonation
// transcriber to transcribe between text and phoneme
// parser to parse text and handle potential commands

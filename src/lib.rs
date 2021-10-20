#![no_std]
#![forbid(unsafe_code)]

// The main file the synth is in
// first, define some constants

/// default sample rate all voices use
/// Resampling to a different sample rate is possible
pub const DEFAULT_SAMPLE_RATE: usize = 44100;

/// the number of formants to synthesize
pub const NUM_FORMANTS: usize = 12;

/// the number of formants that are voiced, and don't receive noise as input
pub const NUM_VOICED_FORMANTS: usize = 8;

// next, let's make a struct to help storing arrays, and do operations on them

/// Array, containing NUM_FORMANTS elements. Used to store formants
pub struct Array { 
	/// inner array
	pub x: [f32; NUM_FORMANTS] 
}

impl Array {

	/// makes a new Array from a given array 
	pub fn new(arr: [f32; NUM_FORMANTS]) -> Self {
		Self { x: arr }
	}

	// and arithmatic
	// not using the OP traits here to keep it simple

	/// adds two arrays together
	pub fn add(self, other: Self) -> Self {
		let mut res = self;
		for i in 0..NUM_FORMANTS {
			res.x[i] += other.x[i];
		}
		res
	}

	/// subtracts an array from another
	pub fn sub(self, other: Self) -> Self {
		let mut res = self;
		for i in 0..NUM_FORMANTS {
			res.x[i] -= other.x[i];
		}
		res
	}

	/// multiplies two arrays together
	pub fn mul(self, other: Self) -> Self {
		let mut res = self;
		for i in 0..NUM_FORMANTS {
			res.x[i] *= other.x[i];
		}
		res
	}

	/// divides one array with another
	pub fn div(self, other: Self) -> Self {
		let mut res = self;
		for i in 0..NUM_FORMANTS {
			res.x[i] /= other.x[i];
		}
		res
	}

	/// sums all elements in an array together
	pub fn sum(self) -> f32 {
		self.x.iter().sum()
	}

	/// blend two arrays, based on some blend value
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
pub fn random_float(state: &mut u32) -> f32 {
    // here we change the state with a regular integer rng
    // This is the lehmer random number generator: https://en.wikipedia.org/wiki/Lehmer_random_number_generator
    // 16807 here is a magic number. In theory this could be any coprime, but there are some numbers that work better
    // 48271 is also such a number
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
pub struct SynthesisElem {

	/// base frequency, normalized to sample rate
	pub frequency: f32,

	/// amplitude
	pub amplitdue: f32,

	/// formant frequencies, normalized to sample rate
	pub formant_frequencies: Array,

	/// formant bandwidths, normalized to sample rate
	pub formant_bandwidths: Array,

	/// formant amplitudes, needs to sum up to one
	pub formant_amplitudes: Array,

	/// antiresonator frequency, normalized to sample rate
	pub antiresonator_frequency: f32,

	/// antiresonator bandwidth
	pub antiresonator_bandwidth: f32,

	/// antiresonator amplitude
	pub antiresonator_amplitude: f32,

	/// voice softness, 0 is saw, 1 is sine 
	pub softness: f32,

}

// next, make some functions for the element
// we want to make one from some sample rate, make one with the given sample rate, and blend them

// next we'll want to synthesize some audio
// for that, we'll use an iterator
// it keeps track of the filter states, and the underlying iterator to get synthesis elements from
// if iterators aren't available in the language you are porting to, use a function to get the next item from some state instead
pub struct Synthesize<T: Iterator<Item=SynthesisElem>> {

	/// underlying iterator
	iter: T,

	/// filter state a
	filter_state_a: Array,

	/// filter state b
	filter_state_b: Array,

	/// antiresonator state a
	antiresonator_state_a: f32,

	/// antiresonator state b
	antiresonator_state_b: f32,

	/// phase for the current pulse
	phase: f32,

	/// noise state
	seed: u32,

}

// next up, implement iterator for the synthesizer, which takes care of synthesizing sound (in samples) from synthesis elements

// We also want to jitter all frequencies a bit for more realism, so let's do that next

// Here's how it will work
// synthesizer iterator to generate sound
// jitter iterator to add randomness to the frequencies
// sequencer iterator to blend phonemes
// intonator to add intonation
// transcriber to transcribe between text and phoneme
// parser to parse text and handle potential commands


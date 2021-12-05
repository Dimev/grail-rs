#![no_std]
#![forbid(unsafe_code)]

// we'll want to implement these for arrays
use core::ops::{Add, Div, Mul, Sub};

// we'll want to allow voices to be used from this library
pub mod voices;

// The main file the synth is in
// first, define some constants

/// default sample rate all voices use
/// Resampling to a different sample rate is possible
pub const DEFAULT_SAMPLE_RATE: u32 = 44100;

/// the number of formants to synthesize
pub const NUM_FORMANTS: usize = 12;

/// the number of formants that are voiced, and don't receive noise as input
pub const NUM_VOICED_FORMANTS: usize = 8;

// and some arithmatic functions
// these are approximations to help speed things up
// hyperbolic tangent, x is multiplied by pi
fn tan_approx(x: f32) -> f32 {

	// tan(x) = sin(x) / cos(x)
	// we can approximate sin and x with the bhaskara I approximation quite well
	// which is 16x(pi - x) / 5pi^2 - 4x(pi - x) for sin
	// if we fill it in, multiply pi by and rewrite it, we get this:
	((1.0 - x) * x * (5.0 - 4.0 * (x + 0.5) * (0.5 - x))) / ((x + 0.5) * (5.0 - 4.0 * (1.0 - x) * x) * (0.5 - x))

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

    /// makes a new array and fills it with a single element
    #[inline]
    pub fn splat(val: f32) -> Self {
        Self {
            x: [val; NUM_FORMANTS],
        }
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

	/// hyperbolic tangent approximation
	#[inline]
	pub fn tan_approx(self) -> Self {
		let mut res = self;
		for i in 0..NUM_FORMANTS {
			res.x[i] = tan_approx(res.x[i])
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

// and, a helper function to do random number generation

/// generates a random float, and changes the state after doing so
#[inline]
fn random_f32(state: &mut u32) -> f32 {
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

// next up, let's go to the audio part
// we'll want a way to represent what to synthesize

/// synthesis element, describes what to synthesize
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
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
    /// make a new synthesis element. For unit gain, formant_amp should sum up to 1
	pub fn new(
        sample_rate: u32,
        frequency: f32,
        formant_freq: [f32; NUM_FORMANTS],
        formant_bw: [f32; NUM_FORMANTS],
        formant_amp: [f32; NUM_FORMANTS],
        nasal_freq: f32,
        nasal_bw: f32,
        nasal_amp: f32,
        softness: f32,
    ) -> Self {
        Self {
            frequency: frequency / sample_rate as f32,
            formant_freq: Array::new(formant_freq) / Array::splat(sample_rate as f32),
            formant_bw: Array::new(formant_bw) / Array::splat(sample_rate as f32),
            formant_amp: Array::new(formant_amp),
            nasal_freq: nasal_freq / sample_rate as f32,
            nasal_bw: nasal_bw / sample_rate as f32,
            nasal_amp,
            softness,
        }
    }

    /// make a new one with the default sample rate, and unit gain
    pub fn new_phoneme(
        formant_freq: [f32; NUM_FORMANTS],
        formant_bw: [f32; NUM_FORMANTS],
        formant_amp: [f32; NUM_FORMANTS],
        nasal_freq: f32,
        nasal_bw: f32,
        nasal_amp: f32,
        softness: f32,
    ) -> Self {
        Self {
            frequency: 0.0,
            formant_freq: Array::new(formant_freq) / Array::splat(DEFAULT_SAMPLE_RATE as f32),
            formant_bw: Array::new(formant_bw) / Array::splat(DEFAULT_SAMPLE_RATE as f32),
            // divide it by the sum of the entire amplitudes, that way we get unit gain
            formant_amp: Array::new(formant_amp) / Array::splat(Array::new(formant_amp).sum()),
            nasal_freq: nasal_freq / DEFAULT_SAMPLE_RATE as f32,
            nasal_bw: nasal_bw / DEFAULT_SAMPLE_RATE as f32,
            nasal_amp,
            softness,
        }
    }
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

        // get the new frequency
        let formant_freq = self.formant_freq * Array::splat(scale);

        // drop all formants above nyquist
        let mut formant_amp = self.formant_amp;

        for (amp, freq) in formant_amp.x.iter_mut().zip(formant_freq.x) {
            if freq > 0.5 {
                *amp = 0.0;
            }
        }

        Self {
            frequency: self.frequency * scale,
            formant_freq: self.formant_freq * Array::splat(scale),
            formant_bw: self.formant_bw * Array::splat(scale),
            nasal_freq: self.nasal_freq * scale,
            nasal_bw: self.nasal_bw * scale,
            formant_amp,
            ..self // this means fill in the rest of the struct with self
        }
    }

    /// copy it with a different frequency
    /// frequency is already divided by the sample rate here
    #[inline]
    pub fn copy_with_frequency(self, frequency: f32) -> Self {
        Self { frequency, ..self }
    }

    /// copy it without any sound
    #[inline]
    pub fn copy_silent(self) -> Self {
        Self {
            formant_amp: Array::splat(0.0),
            ..self
        }
    }
}

// next we'll want to synthesize some audio
// for that, we'll use an iterator
// it keeps track of the filter states, and the underlying iterator to get synthesis elements from
// if iterators aren't available in the language you are porting to, use a function to get the next item from some state instead
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
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

    fn next(&mut self) -> Option<Self::Item> {
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
        let inp = Array::new({
            // fill it with noise
            let mut arr = [noise; NUM_FORMANTS];

            // and set the first n to the pulse
            for elem in arr.iter_mut().take(NUM_VOICED_FORMANTS) {
                *elem = pulse;
            }

			// TODO: FIGURE OUT PROPER LOUDNESS FOR THE FILTER

            // and return the array to put it in x
            arr
        });

        // make sure it's loud enough
        let x = inp * elem.formant_amp;

		// now, apply the parallel bandpass filter
		// this is a State Variable Filter, from here: https://cytomic.com/files/dsp/SvfLinearTrapOptimised2.pdf
		// first, the parameters
		let g = elem.formant_freq.tan_approx();

		// stuff needed to make the filter parameters
		// k = 1 / Q, and bandwidth = frequency / Q, so rewrite it to get k from the bandwidth and freq
		let k = elem.formant_bw  / elem.formant_freq;
		let a1 = Array::splat(1.0) / (Array::splat(1.0) + g * (g + k));
		let a2 = g * a1;

		// process the filter
		let v1 = a1 * self.formant_state_a + a2 * (x - self.formant_state_b);
		let v2 = self.formant_state_b + g * v1;

		// update the state
		self.formant_state_a = Array::splat(2.0) * v1 - self.formant_state_a;
		self.formant_state_b = Array::splat(2.0) * v2 - self.formant_state_b;

		// we're interested in the bandpass result here
		// which is just v1
		let r = v1.sum();

		// now, do the same to get the notch filter
		let g = tan_approx(elem.nasal_freq);

		// parameters
		let k = elem.nasal_bw / elem.nasal_freq;
		let a1 = 1.0 / (1.0 + g * (g + k));
		let a2 = g * a1;

		// process
		let v1 = a1 * self.nasal_state_a + a2 * (r - self.nasal_state_b);
		let v2 = self.nasal_state_b + g * v1;

		// update
		self.nasal_state_a = 2.0 * v1 - self.nasal_state_a;
		self.nasal_state_b = 2.0 * v2 - self.nasal_state_b;

		// and the notch result, which is also the final result
		Some(r - k * v1 * elem.nasal_amp)
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
            phase: 0.5,
            seed: 0,
        }
    }
}

// implement it for anything that can become the right iterator
impl<T> IntoSynthesize for T where T: IntoIterator<Item = SynthesisElem> + Sized {}

// that's it, sound synthesis done
// before we continue, we'd like to set up the internal represenation for voices
// a voice consists of a number of synthesis elements, each assigned to a phoneme
// a phoneme is the smallest sound in speech, so we can use that for the internal representation nicely
// the downside is that there are quite a few

// first, set up the enum for all phonemes
// TODO: IPA or some reduced set?
// reducet set makes it easier to make voices
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub enum Phoneme {
    Silence, // Generic silence
    A,       // a
}

// next up, a voice storage
// this is not a full voice, but instead all phonemes, so it's easier to pass around
// we won't make a constructor for it due to it not really being needed
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct VoiceStorage {
    pub silence: SynthesisElem,
    pub a: SynthesisElem,
}

impl VoiceStorage {
    /// retreive a synthesis elem based on the given phoneme
    pub fn get(self, phoneme: Phoneme) -> SynthesisElem {
        match phoneme {
            Phoneme::Silence => self.silence,
            Phoneme::A => self.a,
        }
    }

    /// run a function on all phonemes
    pub fn map(&mut self, func: fn(Phoneme, &mut SynthesisElem)) {
        func(Phoneme::Silence, &mut self.silence);
        func(Phoneme::A, &mut self.a);
    }
}
// and next, the full voice
// which is just the voice storage + extra parameters for intonation

/// A voice containing all needed parameters to synthesize sound from some given phonemes
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct Voice {
    /// sample rate this voice is at
    pub sample_rate: u32,

    /// phonemes, to generate sound
    pub phonemes: VoiceStorage,

    /// frequency at which to jitter things, to improve voice naturalness
    pub jitter_frequency: f32,

    /// how much to jitter the base frequency
    pub jitter_delta_frequency: f32,

    /// how much to jitter the formant frequencies
    pub jitter_delta_formant_frequency: f32,

    /// how much to jitter the formant amplitudes
    pub jitter_delta_amplitude: f32,
}

// We also want to jitter all frequencies a bit for more realism, so let's do that next

// first, we want to make a few structs to help with generating noise
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
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
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
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
        self.current * Array::splat(1.0 - self.phase) + self.next * Array::splat(self.phase)
    }
}

// now we can make our jitter work, as getting random numbers is now easier
// all frequencies are in normalized form, so 1.0 is the sample frequency
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct Jitter<T: Iterator<Item = SynthesisElem>> {
    /// underlying iterator
    iter: T,

    /// noise for the frequency
    freq_noise: ValueNoise,

    /// noise for the formant frequency
    formant_freq_noise: ArrayValueNoise,

    /// noise for the formant amplitude
    formant_amp_noise: ArrayValueNoise,

    /// nasal frequency
    nasal_freq_noise: ValueNoise,

    /// nasal amplitude
    nasal_amp_noise: ValueNoise,

    /// noise frequency
    frequency: f32,

    /// frequency deviation
    delta_frequency: f32,

    /// formant deviation, also includes antiresonator/nasal
    delta_formant_freq: f32,

    /// amplitude deviation, also includes antiresonator/nasal
    delta_amplitude: f32,
}

impl<T: Iterator<Item = SynthesisElem>> Iterator for Jitter<T> {
    type Item = SynthesisElem;

    fn next(&mut self) -> Option<Self::Item> {
        // get the next element from the underlying iterator
        let mut elem = self.iter.next()?;

        // gather all next noises
        let freq = self.freq_noise.next(self.frequency);
        let formant_freq = self.formant_freq_noise.next(self.frequency);
        let formant_amp = self.formant_amp_noise.next(self.frequency);
        let nasal_freq = self.nasal_freq_noise.next(self.frequency);
        let nasal_amp = self.nasal_amp_noise.next(self.frequency);

        // change them in the element
        elem.frequency += freq * self.delta_frequency;
        elem.formant_freq =
            elem.formant_freq + formant_freq * Array::splat(self.delta_formant_freq);
        // we don't want it to get *louder*, so make sure it only becomes softer by doing (1 + [-1, 1]) / 2, which results in [0, 1]
        // we'll then multiply it by the appropriate amplitude so we can't end up with negative amplitudes for some sounds
        let formant_amp_delta =
            (formant_amp + Array::splat(1.0)) * Array::splat(0.5 * self.delta_amplitude);

        // multiplier is 1 - x, so that it doesn't become very soft
        let formant_amp_mul = Array::splat(1.0) - formant_amp_delta;
        elem.formant_amp = elem.formant_amp * formant_amp_mul;

        // just the nasal frequency passing by
        elem.nasal_freq += nasal_freq * self.delta_formant_freq;

        // we'll want the same for the nasal amplitude
        let nasal_amp_delta = (nasal_amp + 1.0) * (self.delta_amplitude * 0.5);
        let nasal_amp_mul = 1.0 - nasal_amp_delta;
        elem.nasal_amp *= nasal_amp_mul;

        // and return the modified element
        Some(elem)
    }
}

// and we want to be able to easily make the jitter iterator
pub trait IntoJitter
where
    Self: IntoIterator<Item = SynthesisElem> + Sized,
{
    /// creates a new synthesizer from this iterator
    fn jitter(self, mut seed: u32, voice: Voice) -> Jitter<Self::IntoIter> {
        Jitter {
            iter: self.into_iter(),
            freq_noise: ValueNoise::new(&mut seed),
            formant_freq_noise: ArrayValueNoise::new(&mut seed),
            formant_amp_noise: ArrayValueNoise::new(&mut seed),
            nasal_freq_noise: ValueNoise::new(&mut seed),
            nasal_amp_noise: ValueNoise::new(&mut seed),
            frequency: voice.jitter_frequency,
            delta_frequency: voice.jitter_delta_frequency,
            delta_formant_freq: voice.jitter_delta_formant_frequency,
            delta_amplitude: voice.jitter_delta_amplitude,
        }
    }
}

// implement it for anything that can become the right iterator
impl<T> IntoJitter for T where T: IntoIterator<Item = SynthesisElem> + Sized {}

// we now have a way to synthesize sound, and add random variations to it.
// However, generating the induvidual samples is kinda a hassle to do, so it would be nicer if we can give each synthesis element a length
// and then generate the right sequence from that
// so, we'll create a sequencer that does this

// for this, we'll first need a struct to help with adding the time
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct SequenceElem {
    /// the synthesis element
    pub elem: SynthesisElem,

    /// time this element lasts for
    pub length: f32,

    /// time the blending lasts for
    pub blend_length: f32,
}

impl SequenceElem {
    /// make a new element
    pub fn new(elem: SynthesisElem, length: f32, blend_length: f32) -> Self {
        Self {
            elem,
            length,
            blend_length,
        }
    }
}

/// Sequencer, given a time and blend time, it generates the right amount of samples
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct Sequencer<T: Iterator<Item = SequenceElem>> {
    /// underlying iterator
    iter: T,

    /// current element
    cur_elem: Option<SequenceElem>,

    /// next element
    next_elem: Option<SequenceElem>,

    /// time remaining for this element
    time: f32,

    /// sample time, how long a sample lasts (1 / sample rate)
    delta_time: f32,
}

impl<T: Iterator<Item = SequenceElem>> Iterator for Sequencer<T> {
    type Item = SynthesisElem;

    fn next(&mut self) -> Option<Self::Item> {
        // decrease the amount of remaining time
        self.time -= self.delta_time;

        // if this is now below 0, we go to the next pair
        if self.time < 0.0 {
            // figure out what to do next
            match (self.cur_elem, self.next_elem) {
                // we have both, get a new one
                (Some(_), Some(a)) => {
                    self.cur_elem = self.next_elem;
                    self.next_elem = self.iter.next();

                    // set the time
                    self.time = a.length;
                }
                // we have none, fetch new ones
                (None, None) => {
                    self.cur_elem = self.iter.next();
                    self.next_elem = self.iter.next();

                    // if we have the current one, set the time
                    if let Some(a) = self.cur_elem {
                        self.time = a.length;
                    }
                }
                // for the rest, we can simply exit early
                _ => return None,
            }
        }

        // and match on what to do
        match (self.cur_elem, self.next_elem) {
            // both elements, blend to the next one
            (Some(a), Some(b)) => {
                // get the blend amount
                let alpha = (self.time / a.blend_length).min(1.0);

                // and blend the 2, because alpha goes from 1 to 0, we need to blend in the other order
                Some(b.elem.blend(a.elem, alpha))
            }

            // only the first one, blend to the end
            (Some(a), None) => {
                // get the blend amount
                let alpha = (self.time / a.blend_length).min(1.0);

                // and blend with a silent one
                Some(a.elem.copy_silent().blend(a.elem, alpha))
            }

            // nothing else, return none
            _ => None,
        }
    }
}

// and implement an easy way to get the iterator
pub trait IntoSequencer
where
    Self: IntoIterator<Item = SequenceElem> + Sized,
{
    /// creates a new sequencer, with the given sample rate
    fn sequence(self, sample_rate: u32) -> Sequencer<Self::IntoIter> {
        Sequencer {
            iter: self.into_iter(),
            delta_time: 1.0 / sample_rate as f32,
            cur_elem: None,
            next_elem: None,
            time: 0.0,
        }
    }
}

// implement it for anything that can become the right iterator
impl<T> IntoSequencer for T where T: IntoIterator<Item = SequenceElem> + Sized {}

// next up, we'll want to go from time + phoneme info to a sequence element, so let's do that
// first, we'll want a new struct to also store timing info with phonemes
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct PhonemeElem {
    /// the phoneme
    pub phoneme: Phoneme,

    /// total length
    pub length: f32,

    /// length of blending
    pub blend_length: f32,

    /// the base frequency, normalized, so 1.0 is the sample frequency
    pub frequency: f32,
}

// and we'll want to make the selector next.
// this simply selects the right synthesis elem from a voice
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct Selector<T: Iterator<Item = PhonemeElem>> {
    /// underlying iterator
    iter: T,

    /// underlying voice storage to get voice data from
    voice_storage: VoiceStorage,
}

impl<T: Iterator<Item = PhonemeElem>> Iterator for Selector<T> {
    type Item = SequenceElem;

    fn next(&mut self) -> Option<Self::Item> {
        // get the next item if we can
        let phoneme = self.iter.next()?;

        // get the right synthesis elem for this phoneme
        let elem = self.voice_storage.get(phoneme.phoneme);

        // and put it in a sequence element
        Some(SequenceElem::new(
            elem.copy_with_frequency(phoneme.frequency),
            phoneme.length,
            phoneme.blend_length,
        ))
    }
}

pub trait IntoSelector
where
    Self: IntoIterator<Item = PhonemeElem> + Sized,
{
    /// creates a selector from the given voice
    fn select(self, voice: Voice) -> Selector<Self::IntoIter> {
        Selector {
            iter: self.into_iter(),
            voice_storage: voice.phonemes,
        }
    }
}

// implement it for anything that can become the right iterator
impl<T> IntoSelector for T where T: IntoIterator<Item = PhonemeElem> + Sized {}

// now, we need to do some more complex stuff again.
// so far we got most of the sound generating "backend" done, now time for the "frontend"
// this needs to take in text and convert it into phonemes + timing.
// we'll first want to define a language, which contains all rules needed for this translation
// TODO: how?

// Here's how it will work
// synthesizer iterator to generate sound
// jitter iterator to add randomness to the frequencies
// sequencer iterator to blend phonemes
// intonator to add intonation
// transcriber to transcribe between text and phoneme
// parser to parse text and handle potential commands

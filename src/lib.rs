#![no_std]
#![forbid(unsafe_code)]

// TODO: move phoneme related stuff into phoneme, and language related stuff into either language or transcribe
// TODO: consider const generics (when done ofc)?
// TODO: make most of the order easy to read, so keep the explanation
// TODO: maybe have a CPAL feature to allow easy playback?
// TODO: move back to svf filters, as they are easier and probably result in better audio

// we'll want to allow voices to be used from this library
pub mod voices;

// and languages
pub mod languages;

// The main file the synth is in
// first, define some constants

/// default sample rate all voices use
/// Resampling to a different sample rate is possible
pub const DEFAULT_SAMPLE_RATE: f32 = 44100.0;

/// the number of formants to synthesize
pub const NUM_FORMANTS: usize = 8;

// we'll want to implement these for arrays
use core::ops::{Add, ControlFlow, Div, Mul, Neg, Sub};

// We'll need some helper functions
// random number generation

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
pub struct Array([f32; NUM_FORMANTS]);

impl Array {
    /// makes a new Array from a given array
    #[inline]
    pub fn new(arr: [f32; NUM_FORMANTS]) -> Self {
        Self(arr)
    }

    /// make a new array from a given function
    #[inline]
    pub fn from_func<F: FnMut() -> f32>(mut f: F) -> Self {
        Self([(); NUM_FORMANTS].map(|_| f()))
    }

    /// makes a new array and fills it with a single element
    #[inline]
    pub fn splat(val: f32) -> Self {
        Self([val; NUM_FORMANTS])
    }

    /// do something for every value in the array
    #[inline]
    pub fn map<F: Fn(f32) -> f32>(self, f: F) -> Self {
        Self(self.0.map(f))
    }

    /// do something for every value in this array and the other
    #[inline]
    pub fn map2<F: Fn(f32, f32) -> f32>(self, other: Self, f: F) -> Self {
        Self(core::array::from_fn(|i| f(self.0[i], other.0[i])))
    }

    /// sums all elements in an array together
    #[inline]
    pub fn sum(self) -> f32 {
        self.0.iter().sum::<f32>()
    }

    /// take the min of 2 arrays, element wise
    #[inline]
    pub fn min(self, other: Self) -> Self {
        self.map2(other, f32::min)
    }

    /// blend two arrays, based on some blend value
    #[inline]
    pub fn blend(self, other: Self, alpha: f32) -> Self {
        self.map2(other, |a, b| a * (1.0 - alpha) + b * alpha)
    }

    /// blend two arrays, based on some blend array
    #[inline]
    pub fn blend_multiple(self, other: Self, alpha: Array) -> Self {
        self * (Array::splat(1.0) - alpha) + other * alpha
    }

    /// hyperbolic tangent approximation
    #[inline]
    pub fn tan_approx(self) -> Self {
        self.map(|x| tan_approx(x))
    }
}

// and arithmatic
// using the Op  traits to make life easier here, this way we can just do +, - * and /
impl Add for Array {
    type Output = Self;
    /// adds the values in two arrays together
    #[inline]
    fn add(self, other: Self) -> Self {
        self.map2(other, |a, b| a + b)
    }
}

impl Sub for Array {
    type Output = Self;
    /// subtracts the values in an array from another
    #[inline]
    fn sub(self, other: Self) -> Self {
        self.map2(other, |a, b| a - b)
    }
}

impl Mul for Array {
    type Output = Self;
    /// multiplies the values in two arrays together
    #[inline]
    fn mul(self, other: Self) -> Self {
        self.map2(other, |a, b| a * b)
    }
}

impl Div for Array {
    type Output = Self;
    /// divides the values of one array with another
    #[inline]
    fn div(self, other: Self) -> Self {
        self.map2(other, |a, b| a / b)
    }
}

impl Neg for Array {
    type Output = Self;
    /// negates all values in the array
    #[inline]
    fn neg(self) -> Self {
        self.map(|x| -x)
    }
}

// As well as a few utils to do better random generation, we want to make a few structs to help with generating noise
/// Value noise
/// This noise works by generating two values, and interpolates between them to generate the noise
/// once it has gone too far, it generates a new next point to interpolate to
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct ValueNoise {
    current: f32,
    next: f32,
    phase: f32,
    state: u32,
}

impl ValueNoise {
    /// make a new value noise generator, from the given seed
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

    /// generate the next value, and update the internal state
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
/// Value noise, for arrays
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct ArrayValueNoise {
    current: Array,
    next: Array,
    phase: f32,
    state: u32,
}

impl ArrayValueNoise {
    /// generate a new value noise generator, from the given seed
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

    /// generate the next value, and update the internal state
    pub fn next(&mut self, increment: f32) -> Array {
        // increment the state
        self.phase += increment;

        // wrap it around if needed
        if self.phase > 1.0 {
            self.phase -= 1.0;

            // also update the noise
            self.current = self.next;

            // next noise step
            self.next = Array::from_func(|| random_f32(&mut self.state));
        }

        // and blend between the current and next
        self.current * Array::splat(1.0 - self.phase) + self.next * Array::splat(self.phase)
    }
}

// next up, let's go to the audio part
// we'll want a way to represent what to synthesize

/// Synthesis element, describes what to synthesize
/// This describes the frequency and formants to synthesize
/// All frequency values are normalized to 0-1, where 0 is 0 hz, and 1 is the sample frequency
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct SynthesisElem {
    /// base frequency, normalized to sample rate
    pub frequency: f32,

    /// formant frequencies, normalized to sample rate
    pub formant_freq: Array,

    /// formant bandwidth frequencies, normalized to sample rate
    pub formant_bw: Array,

    /// formant smoothness, how much the higher frequencies are filtered out
    pub formant_smooth: Array,

    /// how breathy each formant is. 0 means fully voiced, 1 means full breath
    pub formant_breath: Array,

    /// formant turbulence, affects how much noise is blended through when the glottis is open
    pub formant_turb: Array,

    /// formant amplitudes. If these sum up to one, the output amplitude will also be one
    pub formant_amp: Array,
}

// next, make some functions for the element
// we want to make one from some sample rate, make one with the given sample rate, and blend them
impl SynthesisElem {
    /// make a new synthesis element. For unit gain, formant_amp should sum up to 1
    pub fn new(
        sample_rate: f32,
        frequency: f32,
        formant_freq: [f32; NUM_FORMANTS],
        formant_smooth: [f32; NUM_FORMANTS],
        formant_bw: [f32; NUM_FORMANTS],
        formant_breath: [f32; NUM_FORMANTS],
        formant_turb: [f32; NUM_FORMANTS],
        formant_amp: [f32; NUM_FORMANTS],
    ) -> Self {
        // make a new element, and then resample it to the appropriate sample rate
        Self {
            frequency: frequency,
            formant_freq: Array::new(formant_freq),
            formant_bw: Array::new(formant_bw),
            formant_smooth: Array::new(formant_smooth),
            formant_breath: Array::new(formant_breath),
            formant_turb: Array::new(formant_turb),
            formant_amp: Array::new(formant_amp),
        }
        .resample(1.0, sample_rate)
    }

    /// create a new silent item
    pub fn silent() -> Self {
        Self {
            frequency: 0.25,
            formant_freq: Array::splat(0.25),
            formant_bw: Array::splat(0.25),
            formant_smooth: Array::splat(0.25),
            formant_breath: Array::splat(0.0),
            formant_turb: Array::splat(0.0),
            formant_amp: Array::splat(0.0),
        }
    }

    /// Make a new one with the default sample rate
    /// Also ensure that the formant amplitudes sum up to 1 to get unit gain
    pub fn new_phoneme(
        formant_freq: [f32; NUM_FORMANTS],
        formant_bw: [f32; NUM_FORMANTS],
        formant_smooth: [f32; NUM_FORMANTS],
        formant_turb: [f32; NUM_FORMANTS],
        formant_breath: [f32; NUM_FORMANTS],
        formant_amp: [f32; NUM_FORMANTS],
    ) -> Self {
        Self {
            frequency: 0.0,
            formant_freq: Array::new(formant_freq),
            formant_bw: Array::new(formant_bw),
            formant_smooth: Array::new(formant_smooth),
            formant_breath: Array::new(formant_breath),
            formant_turb: Array::new(formant_turb),

            // divide it by the sum of the entire amplitudes, that way we get unit gain
            formant_amp: Array::new(formant_amp) / Array::splat(Array::new(formant_amp).sum()),
        }
        .resample(1.0, DEFAULT_SAMPLE_RATE)
    }
    /// blend between this synthesis element and another one
    #[inline]
    pub fn blend(self, other: Self, alpha: f32) -> Self {
        Self {
            frequency: self.frequency * (1.0 - alpha) + other.frequency * alpha,
            formant_freq: self.formant_freq.blend(other.formant_freq, alpha),
            formant_smooth: self.formant_smooth.blend(other.formant_smooth, alpha),
            formant_bw: self.formant_bw.blend(other.formant_bw, alpha),
            formant_turb: self.formant_turb.blend(other.formant_turb, alpha),
            formant_breath: self.formant_breath.blend(other.formant_breath, alpha),
            formant_amp: self.formant_amp.blend(other.formant_amp, alpha),
        }
    }

    /// resample the synthesis element to a new sample rate
    #[inline]
    pub fn resample(self, old_sample_rate: f32, new_sample_rate: f32) -> Self {
        // scale factor for the sample rate
        let scale = old_sample_rate / new_sample_rate;

        // get the new frequency
        let formant_freq = self.formant_freq * Array::splat(scale);

        Self {
            // make sure it doesn't go above nyquist
            frequency: (self.frequency * scale).min(0.5),
            formant_freq: (self.formant_freq * Array::splat(scale)).min(Array::splat(0.5)),
            formant_bw: self.formant_bw * Array::splat(scale),

            // drop all values above nyquist
            formant_amp: self
                .formant_amp
                .map2(formant_freq, |amp, freq| if freq > 0.5 { 0.0 } else { amp }),

            // leave the rest intact
            ..self
        }
    }

    /// copy it with a different frequency
    /// frequency is already divided by the sample rate here
    #[inline]
    pub fn copy_with_frequency(self, frequency: f32) -> Self {
        Self {
            frequency: frequency.min(0.5),
            ..self
        }
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

/// Synthesizer for synthesizing actual audio samples from synthesis elements
/// this is created by calling .synthesize() on an iterator that produces synthesis elements
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct Synthesize<T: Iterator<Item = SynthesisElem>> {
    /// underlying iterator
    iter: T,

    /// phase of the carrier
    phase: f32,

    /// lowpass filter state
    filter_state_a: Array,

    /// svf filter state 1 (ic1eq)
    filter_state_b: Array,

    /// svf filter statee 2 (ic2eq)
    filter_state_c: Array,

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

        // generate an anti-aliased saw wave
        // polyblep offset, to do antialiasing
        let polyblep = if self.phase < elem.frequency {
            // if we're at the first sample, smooth it a bit
            let t = self.phase / elem.frequency;
            2.0 * t - (t * t) - 1.0
        } else if self.phase > (1.0 - elem.frequency) {
            // same for last sample
            let t = (self.phase - 1.0) / elem.frequency;
            (t * t) + 2.0 * t + 1.0
        } else {
            // otherwise, no smoothing needed
            0.0
        };

        // saw wave
        let saw_wave = (2.0 * self.phase - 1.0) - polyblep;

        // increment phase
        self.phase += elem.frequency;

        // wrap around, as this can't be above nyquist we can avoid fract
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        // generate some noise
        let noise = Array::splat(random_f32(&mut self.seed));

        // apply a low pass filter, single pole
        self.filter_state_a = self.filter_state_a
            + (Array::splat(1.0) - elem.formant_smooth)
                * (Array::splat(saw_wave) - self.filter_state_a);

        // get the result from the filter
        let glottal_wave = self.filter_state_a;

        // blend it with the noise based on the turbulence and breath
        // turbulence is extra noise added when the glottis is open, while breath is always on
        let turbulence_wave = (glottal_wave * (Array::splat(1.0) - noise * elem.formant_turb))
            .blend_multiple(noise, elem.formant_breath);

        // apply amplitude and scale it
        // makes sure it's the right amplitude to not make the filter go out of the [-1, 1] range
        // TODO: how
        let v0 = turbulence_wave * elem.formant_amp;

        // state variable filter
        // https://cytomic.com/files/dsp/SvfLinearTrapOptimised2.pdf
        // set the parameters
        let g = elem.formant_freq.tan_approx();

        // k = 1 / Q, and Q = f_r / delta_f, where f_r is the resonant frequency, and delta_f is the bandwidth
        let k = elem.formant_bw / elem.formant_freq;

        let a1 = Array::splat(1.0) / (Array::splat(1.0) + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;

        // step the filter forwards to get the next state
        let v3 = v0 - self.filter_state_c;
        let v1 = a1 * self.filter_state_b + a2 * v3;
        let v2 = self.filter_state_c + a2 * self.filter_state_b + a3 * v3;

        // update actual state
        self.filter_state_b = Array::splat(2.0) * v1 - self.filter_state_b;
        self.filter_state_c = Array::splat(2.0) * v2 - self.filter_state_c;

        // and the bandpass result
        let res = v1.sum() * 0.5;

        // and return the wave, scaled by amplitude
        Some(res)
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
            phase: 0.0,
            filter_state_a: Array::splat(0.0),
            filter_state_b: Array::splat(0.0),
            filter_state_c: Array::splat(0.0),
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
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Phoneme {
    /// Silence, somewhat special as blending the phoneme to this will only blend the amplitude
    Silence,
    A, // a
    E, // e
}

// next up, a voice storage
// this is not a full voice, but instead all phonemes, so it's easier to pass around
// we won't make a constructor for it due to it not really being needed
/// Stores a synthesis element for each phoneme
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct VoiceStorage {
    pub a: SynthesisElem,
    pub e: SynthesisElem,
}

impl VoiceStorage {
    /// retreive a synthesis elem based on the given phoneme
    pub fn get(self, phoneme: Phoneme) -> Option<SynthesisElem> {
        match phoneme {
            Phoneme::Silence => None,
            Phoneme::A => Some(self.a),
            Phoneme::E => Some(self.e),
        }
    }

    /// run a function on all phonemes
    pub fn for_all(&mut self, func: fn(Phoneme, &mut SynthesisElem)) {
        func(Phoneme::A, &mut self.a);
        func(Phoneme::E, &mut self.e);
    }
}

// and next, the full voice
// which is just the voice storage + extra parameters for intonation

/// A voice containing all needed parameters to synthesize sound from some given phonemes
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct Voice {
    /// sample rate this voice is at
    pub sample_rate: f32,

    /// phonemes, to generate sound
    pub phonemes: VoiceStorage,

    /// center frequency for the voice
    pub center_frequency: f32,

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

    /// noise frequency
    frequency: f32,

    /// frequency deviation
    delta_frequency: f32,

    /// formant deviation
    delta_formant_freq: f32,

    /// amplitude deviation
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
    /// Some if there, None if silent
    pub elem: Option<SynthesisElem>,

    /// time this element lasts for
    pub length: f32,

    /// time the blending lasts for
    pub blend_length: f32,
}

impl SequenceElem {
    /// make a new element
    pub fn new(elem: Option<SynthesisElem>, length: f32, blend_length: f32) -> Self {
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
                    self.time += a.length;
                }
                // we have none, fetch new ones
                (None, None) => {
                    self.cur_elem = self.iter.next();
                    self.next_elem = self.iter.next();

                    // if we have the current one, set the time
                    if let Some(a) = self.cur_elem {
                        self.time += a.length;
                    }
                }
                // for the rest, we can simply exit early
                _ => return None,
            }
        }

        // and match on what to do
        match (
            self.cur_elem,
            self.cur_elem.and_then(|x| x.elem),
            self.next_elem.and_then(|x| x.elem),
        ) {
            // both elements, all are on
            (Some(a), Some(b), Some(c)) => {
                // get the blend amount
                let alpha = (self.time / a.blend_length).min(1.0);

                // and blend the 2, because alpha goes from 1 to 0, we need to blend in the other order
                Some(c.blend(b, alpha))
            }

            // only the first one, blend to silence
            (Some(a), Some(b), None) => {
                // get the blend amount
                let alpha = (self.time / a.blend_length).min(1.0);

                // and blend with a silent one
                Some(b.copy_silent().blend(b, alpha))
            }

            // only the first one, blend from silence
            (Some(a), None, Some(c)) => {
                // get the blend amount
                let alpha = (self.time / a.blend_length).min(1.0);

                // and blend with a silent one
                Some(c.blend(c.copy_silent(), alpha))
            }

            // both silent
            (Some(_), None, None) => {
                // just return silence
                Some(SynthesisElem::silent())
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
    fn sequence(self, sample_rate: f32) -> Sequencer<Self::IntoIter> {
        Sequencer {
            iter: self.into_iter(),
            delta_time: 1.0 / sample_rate,
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
// as well as makes sure a silence is blended correctly
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

        // get the right phoneme, or none if it's silent.
        // this allows correct blending later on
        let elem = self.voice_storage.get(phoneme.phoneme);

        // and put it in a sequence element
        Some(SequenceElem::new(
            // if there is any, copy it with the right frequency
            elem.map(|x| x.copy_with_frequency(phoneme.frequency)),
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
// let's first make the rules we use for text -> phoneme

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct TranscriptionRule<'a> {
    /// string to compare agains
    pub string: &'a str,

    /// phonemes to generate from this
    pub phonemes: &'a [Phoneme],
}

// now make the actual language, which is just a set of transcription rules
pub struct Language<'a> {
    /// rules for the language to transcribe phonemes
    pub rules: &'a [TranscriptionRule<'a>],

    /// whether the language is case-sensitive
    pub case_sensitive: bool,
}

// next up, the intonator.
// this will add intonation to any phoneme sequence
pub struct Intonator<T: Iterator<Item = Phoneme>> {
    /// underlying iterator
    iter: T,

    /// center frequency for the voice
    center_frequency: f32,
}

impl<T: Iterator<Item = Phoneme>> Iterator for Intonator<T> {
    type Item = PhonemeElem;
    fn next(&mut self) -> Option<Self::Item> {
        let phon = self.iter.next()?;

        // TODO: apply intonation

        // TODO: speaking rate

        // TODO: give certain phonemes a length

        Some(PhonemeElem {
            phoneme: phon,
            length: 0.5,
            blend_length: 0.5,
            frequency: self.center_frequency,
        })
    }
}

pub trait IntoIntonator
where
    Self: IntoIterator<Item = Phoneme> + Sized,
{
    fn intonate(self, language: Language, voice: Voice) -> Intonator<Self::IntoIter> {
        Intonator {
            iter: self.into_iter(),
            center_frequency: voice.center_frequency,
        }
    }
}

impl<T> IntoIntonator for T where T: IntoIterator<Item = Phoneme> + Sized {}

// now we want to convert text into phonemes
// we're going to do this with a find-and-replace ruleset, as defined in language.
// this is assumed to be sorted, so we can binary search with the prefix,
// to figure out the range we need to search in and see if it's too low or too high

pub struct Transcriber<'a, T: Iterator<Item = char>> {
    /// underlying iterator
    iter: T,

    /// ruleset to use
    ruleset: &'a [TranscriptionRule<'a>],

    /// whether we are case sensitive to match
    // TODO: replace this with a better ruleset, as in: each rule becomes lowercase
    case_sensitive: bool,

    /// buffer for the phonemes we have now
    buffer: &'a [Phoneme],
}

// silent buffer
const SILENCE: &'static [Phoneme] = &[Phoneme::Silence];

impl<'a, T: Iterator<Item = char>> Iterator for Transcriber<'a, T> {
    type Item = Phoneme;
    fn next(&mut self) -> Option<Self::Item> {
        // if our buffer is empty, try and make a new one
        if self.buffer.len() == 0 {
            // find the right buffer search range
            // using try fold we can stop once we found the right value
            self.buffer = match self
                .iter
                .by_ref()
                .map(|x| {
                    if self.case_sensitive {
                        x
                    } else {
                        x.to_ascii_lowercase()
                    }
                })
                .enumerate()
                .try_fold(
                    (0, self.ruleset.len()),
                    |(search_min, search_max), (index, character)| {
                        // find the new search range
                        // now that we have a new item, we can reduce the search range
                        // this is binary search, where the left half is where the lower range is lexiographically lower than the current buffer content
                        // because we only get one char at a time, we can assume that the previous N characters were already found and reduced the range
                        // so no need to keep those around anymore
                        let new_min = self.ruleset[search_min..search_max].partition_point(|x| {
                            x.string
                                .chars()
                                .skip(index)
                                .next()
                                .map_or(true, |x| x < character)
                        }) + search_min;

                        // same for the upper range, but now it's lower or equal
                        let new_max = self.ruleset[search_min..search_max].partition_point(|x| {
                            x.string
                                .chars()
                                .skip(index)
                                .next()
                                .map_or(false, |x| x <= character)
                        }) + search_min;

                        // now decide on where to go
                        if new_min + 1 == new_max
                            && self.ruleset[new_min].string.chars().count() == index + 1
                        {
                            // equal if is one bigger than the max, we found a rule
                            // also make sure we have the entire rule, so we don't leave out a bit
                            ControlFlow::Break(self.ruleset[new_min].phonemes)
                        } else if new_min == new_max {
                            // if they are equal, no rule was found, so just emit silence
                            ControlFlow::Break(SILENCE)
                        } else {
                            // otherwise, we are still running
                            ControlFlow::Continue((new_min, new_max))
                        }
                    },
                ) {
                // return the found buffer
                ControlFlow::Break(x) => x,

                // if it was still on continue, the underlying iterator has ended
                // this means we won't find any new rule, so stop
                _ => return None,
            };
        }

        // try and get the first item if we have that
        let result = self.buffer.get(0);

        // remove the first item if we can, and set the buffer to the rest of the remaining buffer
        self.buffer = self.buffer.get(1..).unwrap_or(&[]);

        // return the result
        result.map(|x| *x)
    }
}

pub trait IntoTranscriber
where
    Self: IntoIterator<Item = char> + Sized,
{
    fn transcribe(self, language: Language) -> Transcriber<Self::IntoIter> {
        Transcriber {
            iter: self.into_iter(),
            ruleset: language.rules,
            buffer: SILENCE,
            case_sensitive: language.case_sensitive,
        }
    }
}

impl<T> IntoTranscriber for T where T: IntoIterator<Item = char> + Sized {}

// Here's how it will work
// synthesizer iterator to generate sound
// jitter iterator to add randomness to the frequencies
// sequencer iterator to blend phonemes
// intonator to add intonation
// transcriber to transcribe between text and phoneme
// parser to parse text and handle potential commands

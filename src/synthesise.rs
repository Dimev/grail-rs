use crate::array::*;
use crate::random::*;
use crate::DEFAULT_SAMPLE_RATE;

/// synthesis element, describes what to synthesize
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct SynthesisElem {
    /// base frequency, normalized to sample rate
    pub frequency: f32,

    /// formant frequencies, normalized to sample rate
    pub formant_freq: Array,

    /// formant bandwidths, normalized to sample rate
    pub formant_bw: Array,

    /// formant softness, aka how much of it is lowpassed instead of bandpassed, with 1.0 being max softness
    pub formant_soft: Array,

    /// formant amplitudes. If these sum up to one, the output amplitude will also be one
    pub formant_amp: Array,

    /// how breathy each formant is. 0 means fully voiced, 1 means full breath
    pub formant_breath: Array,
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
        formant_soft: [f32; NUM_FORMANTS],
        formant_amp: [f32; NUM_FORMANTS],
        formant_breath: [f32; NUM_FORMANTS],
    ) -> Self {
        Self {
            frequency: frequency / sample_rate as f32,
            formant_freq: Array::new(formant_freq) / Array::splat(sample_rate as f32),
            formant_bw: Array::new(formant_bw) / Array::splat(sample_rate as f32),
            formant_soft: Array::new(formant_soft),
            formant_amp: Array::new(formant_amp),
            formant_breath: Array::new(formant_breath),
        }
    }

    /// create a new silent item
    pub fn silent() -> Self {
        Self {
            frequency: 0.25,
            formant_freq: Array::splat(0.25),
            formant_bw: Array::splat(0.25),
            formant_soft: Array::splat(0.0),
            formant_amp: Array::splat(0.0),
            formant_breath: Array::splat(0.0),
        }
    }

    /// make a new one with the default sample rate, and unit gain
    pub fn new_phoneme(
        formant_freq: [f32; NUM_FORMANTS],
        formant_bw: [f32; NUM_FORMANTS],
        formant_soft: [f32; NUM_FORMANTS],
        formant_amp: [f32; NUM_FORMANTS],
        formant_breath: [f32; NUM_FORMANTS],
    ) -> Self {
        Self {
            frequency: 0.0,
            formant_freq: Array::new(formant_freq) / Array::splat(DEFAULT_SAMPLE_RATE as f32),
            formant_bw: Array::new(formant_bw) / Array::splat(DEFAULT_SAMPLE_RATE as f32),
            formant_soft: Array::new(formant_soft),
            // divide it by the sum of the entire amplitudes, that way we get unit gain
            formant_amp: Array::new(formant_amp) / Array::splat(Array::new(formant_amp).sum()),
            formant_breath: Array::new(formant_breath),
        }
    }
    /// blend between this synthesis element and another one
    #[inline]
    pub fn blend(self, other: Self, alpha: f32) -> Self {
        Self {
            frequency: self.frequency * (1.0 - alpha) + other.frequency * alpha,
            formant_freq: self.formant_freq.blend(other.formant_freq, alpha),
            formant_bw: self.formant_bw.blend(other.formant_bw, alpha),
            formant_soft: self.formant_soft.blend(other.formant_soft, alpha),
            formant_amp: self.formant_amp.blend(other.formant_amp, alpha),
            formant_breath: self.formant_breath.blend(other.formant_breath, alpha),
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

    /// phase of the carrier
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

        // We're using modified FM synthesis here
		// It's not actual FM synthesis however, it's actually AM synthesis
		// it works by having a carrier wave (cosine) with some exponential curve applied to it, mutliplied by a modulator, which is another cosine
		// the modulator is at the formant frequency, carrier at the base frequency 

        // bandwidth
        let k = 2.8;

        // cosine carrier wave, scaled with the exp
        let voiced_carrier = (Array::splat(k)
            * (Array::splat(self.phase * core::f32::consts::TAU).cos()
            - Array::splat(1.0)))
        .exp();

        // lowpassed noise, as the unvoiced carrier
        let unvoiced_carrier = Array::splat(1.0);

        // true carrier, blended based on how breathy it is
        let carrier = voiced_carrier.blend_multiple(unvoiced_carrier, elem.formant_breath);

        // now get the modulator
        // this is another cosine wave
        // however, to allow smooth frequency sliding, this is a blend between two cosines
        // one with a frequency that is multiple of the carrier, rounded down, and the other rounded up
        // so first calculate the phase for those

        // get the multiple of the carrier the modulator is at
        let multiple = elem.formant_freq / Array::splat(elem.frequency);

        // round down
        let mod_freq_a = multiple.floor();

        // round up
        let mod_freq_b = multiple.floor() + Array::splat(1.0);

        // blend between them
        let mod_blend = multiple.fract();

        // get the actual carrier wave
        let modulator = Array::blend_multiple(
            (Array::splat(self.phase.fract() * core::f32::consts::TAU) * mod_freq_a).cos(),
            (Array::splat(self.phase.fract() * core::f32::consts::TAU) * mod_freq_b).cos(),
            mod_blend,
        );

        // now generate the full wave
        let wave = carrier * Array::blend_multiple(modulator, Array::splat(1.0), elem.formant_soft);

        // increment the phase
        self.phase += elem.frequency;

        // phase rollover
        self.phase = self.phase.fract();

        // and return the wave, scaled by amplitude
        Some((wave * elem.formant_amp).sum())

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
            seed: 0,
        }
    }
}

// implement it for anything that can become the right iterator
impl<T> IntoSynthesize for T where T: IntoIterator<Item = SynthesisElem> + Sized {}

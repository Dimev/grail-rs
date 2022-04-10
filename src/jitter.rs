use crate::synthesise::*;
use crate::random::*;
use crate::array::*;
use crate::voice::*;

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
use crate::synthesise::*;

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
    pub sample_rate: u32,

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

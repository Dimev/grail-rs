// #![no_std]
#![forbid(unsafe_code)]

pub mod array;
pub mod intonate;
pub mod jitter;
pub mod random;
pub mod select;
pub mod sequence;
pub mod synthesise;
pub mod transcribe;
pub mod voice;

pub use crate::array::*;

// TODO: move phoneme related stuff into phoneme, and language related stuff into either language or transcribe
// TODO: consider const generics (when done ofc)?
// TODO: make most of the order easy to read, so keep the explanation

// we'll want to allow voices to be used from this library
pub mod voices;

// and languages
pub mod languages;

// The main file the synth is in
// first, define some constants

/// default sample rate all voices use
/// Resampling to a different sample rate is possible
pub const DEFAULT_SAMPLE_RATE: u32 = 44100;

/// number of phonemes stored during transcription
/// This also effectively limits how many phonemes can be in a transcription rule
pub const PHONEME_BUFFER_SIZE: usize = 64;

/// number of characters stored in the buffer when transcribing
/// this is effectively the maximum rule length
pub const TRANSCRIPTION_BUFFER_SIZE: usize = 64;

// and, a helper function to do random number generation
pub use crate::random::*;

// next up, let's go to the audio part
// we'll want a way to represent what to synthesize

pub use crate::synthesise::*;

// that's it, sound synthesis done
// before we continue, we'd like to set up the internal represenation for voices
// a voice consists of a number of synthesis elements, each assigned to a phoneme
// a phoneme is the smallest sound in speech, so we can use that for the internal representation nicely
// the downside is that there are quite a few
pub use crate::voice::*;

// We also want to jitter all frequencies a bit for more realism, so let's do that next

pub use crate::jitter::*;

// we now have a way to synthesize sound, and add random variations to it.
// However, generating the induvidual samples is kinda a hassle to do, so it would be nicer if we can give each synthesis element a length
// and then generate the right sequence from that
// so, we'll create a sequencer that does this
pub use crate::sequence::*;

// next up, we'll want to go from time + phoneme info to a sequence element, so let's do that
// first, we'll want a new struct to also store timing info with phonemes
pub use crate::select::*;

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

pub use crate::intonate::*;

// now we want to convert text into phonemes
// we're going to do this with a find-and-replace ruleset, as defined in language.
// this is assumed to be sorted, so we can binary search with the prefix,
// to figure out the range we need to search in and see if it's too low or too high
pub use crate::transcribe::*;

// Here's how it will work
// synthesizer iterator to generate sound
// jitter iterator to add randomness to the frequencies
// sequencer iterator to blend phonemes
// intonator to add intonation
// transcriber to transcribe between text and phoneme
// parser to parse text and handle potential commands

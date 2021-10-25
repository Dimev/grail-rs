//! All voices given with grail, along with functions to get them
//! 
use crate::{SynthesisElem, NUM_FORMANTS};

// helper for making phonemes
pub const MKPHON: fn([f32; NUM_FORMANTS], [f32; NUM_FORMANTS], [f32; NUM_FORMANTS], f32, f32, f32, f32) -> SynthesisElem = SynthesisElem::new_phoneme;

// include the voices we made
pub mod generic;

// and use it so we can easily get it
pub use generic::generic;

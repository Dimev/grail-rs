//! All voices given with grail, along with functions to get them
//!
use crate::{SynthesisElem, NUM_FORMANTS};

// helper for making phonemes
// if you're porting this, put this in a seperate file somewhere so you don't include all voices when including a single voice
pub const MKPHON: fn(
    freq: [f32; NUM_FORMANTS],
    bw: [f32; NUM_FORMANTS],
    smooth: [f32; NUM_FORMANTS],
    turb: [f32; NUM_FORMANTS],
    breath: [f32; NUM_FORMANTS],
    amp: [f32; NUM_FORMANTS],
) -> SynthesisElem = SynthesisElem::new_phoneme;

// include the voices we made
pub mod generic;

// and use it so we can easily get it
pub use generic::generic;

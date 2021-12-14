//! languages
use crate::{Language, Phoneme, TranscriptionRule};

pub const fn generic() -> Language<'static> {
    Language {
        case_sensitive: false,
        rules: &[
            TranscriptionRule {
                string: "a",
                phonemes: &[Phoneme::A],
            },
            TranscriptionRule {
                string: "b",
                phonemes: &[Phoneme::Silence],
            },
        ],
    }
}

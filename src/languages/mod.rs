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
                string: "e",
                phonemes: &[Phoneme::E],
            },
            TranscriptionRule {
                string: "i",
                phonemes: &[Phoneme::A],
            },
            TranscriptionRule {
                string: "ii",
                phonemes: &[Phoneme::E, Phoneme::A],
            },
            TranscriptionRule {
                string: "oui",
                phonemes: &[Phoneme::A, Phoneme::E, Phoneme::A],
            },
            TranscriptionRule {
                string: "p",
                phonemes: &[Phoneme::Silence],
            },
        ],
    }
}

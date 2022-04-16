use crate::select::*;
use crate::voice::*;
use crate::*;

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

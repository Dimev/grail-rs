use crate::sequence::*;
use crate::voice::*;

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

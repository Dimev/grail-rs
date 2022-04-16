use crate::*;

pub struct Transcriber<'a, T: Iterator<Item = char>> {
    /// underlying iterator
    iter: T,

    /// ruleset to use
    ruleset: &'a [TranscriptionRule<'a>],

    /// whether we are case sensitive to match
    case_sensitive: bool,

    /// buffer for the phonemes we have now
    buffer: [Phoneme; PHONEME_BUFFER_SIZE],

    /// current size of the buffer
    buffer_size: usize,
}

impl<'a, T: Iterator<Item = char>> Iterator for Transcriber<'a, T> {
    type Item = Phoneme;
    fn next(&mut self) -> Option<Self::Item> {
        // min and max search range
        let mut search_min = 0;
        let mut search_max = self.ruleset.len();

        // buffer to store our text, to see what the current rule is
        let mut text_buffer = [' '; TRANSCRIPTION_BUFFER_SIZE];
        let mut text_buffer_size = 0;

        // loop as long as we need a phoneme
        while self.buffer_size == 0 {
            // try and get an item
            if let Some(character) = self.iter.next() {
                // add it to the buffer if possible
                if text_buffer_size < text_buffer.len() {
                    // make sure it's the right case
                    text_buffer[text_buffer_size] = if self.case_sensitive {
                        character.to_ascii_lowercase()
                    } else {
                        character
                    };
                    text_buffer_size += 1;
                }
            } else {
                // we won't find a new rule, so stop
                break;
            }

            // now that we have a new item, we can reduce the search range
            // this is binary search, where the left half is where the lower range is lexiographically lower than the current buffer content
            let new_min = self.ruleset[search_min..search_max].partition_point(|x| {
                x.string
                    .chars()
                    .take(text_buffer_size)
                    .lt(text_buffer[..text_buffer_size].iter().cloned())
            }) + search_min;

            // same for the upper range, but now it's lower or equal
            let new_max = self.ruleset[search_min..search_max].partition_point(|x| {
                x.string
                    .chars()
                    .take(text_buffer_size)
                    .le(text_buffer[..text_buffer_size].iter().cloned())
            }) + search_min;

            // if the ranges are equal, no rule was found, so insert a silence
            if new_min == new_max && self.buffer_size < self.buffer.len() {
                self.buffer[self.buffer_size] = Phoneme::Silence;
                self.buffer_size += 1;
            } else if new_min + 1 == new_max
                && self.ruleset[new_min] // also make sure that the rule equals our text, so we don't cut it off early
                    .string
                    .chars()
                    .eq(text_buffer[..text_buffer_size].iter().cloned())
            {
                // if it's one, then we found a rule, so add it
                for phoneme in self.ruleset[new_min]
                    .phonemes
                    .iter()
                    .take(self.buffer.len() - self.buffer_size)
                {
                    self.buffer[self.buffer_size] = *phoneme;
                    self.buffer_size += 1;
                }
                // also empty the buffer
                // this is to ensure we can continue parsing if this rule doesn't produce phonemes
                text_buffer_size = 0;
            }

            // and set the range for the next iteration
            search_min = new_min;
            search_max = new_max;
        }

        // if we have items in the phoneme buffer, return one
        if self.buffer_size > 0 {
            // pop the item
            self.buffer_size -= 1;
            Some(self.buffer[self.buffer_size])
        } else {
            None
        }
    }
}

pub trait IntoTranscriber
where
    Self: IntoIterator<Item = char> + Sized,
{
    fn transcribe(self, language: Language) -> Transcriber<Self::IntoIter> {
        Transcriber {
            iter: self.into_iter(),
            ruleset: language.rules,
            buffer: [Phoneme::Silence; PHONEME_BUFFER_SIZE],
            buffer_size: 1,
            case_sensitive: language.case_sensitive,
        }
    }
}

impl<T> IntoTranscriber for T where T: IntoIterator<Item = char> + Sized {}

use crate::synthesise::*;

// for this, we'll first need a struct to help with adding the time
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct SequenceElem {
    /// the synthesis element
    /// Some if there, None if silent
    pub elem: Option<SynthesisElem>,

    /// time this element lasts for
    pub length: f32,

    /// time the blending lasts for
    pub blend_length: f32,
}

impl SequenceElem {
    /// make a new element
    pub fn new(elem: Option<SynthesisElem>, length: f32, blend_length: f32) -> Self {
        Self {
            elem,
            length,
            blend_length,
        }
    }
}

/// Sequencer, given a time and blend time, it generates the right amount of samples
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct Sequencer<T: Iterator<Item = SequenceElem>> {
    /// underlying iterator
    iter: T,

    /// current element
    cur_elem: Option<SequenceElem>,

    /// next element
    next_elem: Option<SequenceElem>,

    /// time remaining for this element
    time: f32,

    /// sample time, how long a sample lasts (1 / sample rate)
    delta_time: f32,
}

impl<T: Iterator<Item = SequenceElem>> Iterator for Sequencer<T> {
    type Item = SynthesisElem;

    fn next(&mut self) -> Option<Self::Item> {
        // decrease the amount of remaining time
        self.time -= self.delta_time;

        // if this is now below 0, we go to the next pair
        if self.time < 0.0 {
            // figure out what to do next
            match (self.cur_elem, self.next_elem) {
                // we have both, get a new one
                (Some(_), Some(a)) => {
                    self.cur_elem = self.next_elem;
                    self.next_elem = self.iter.next();

                    // set the time
                    self.time = a.length;
                }
                // we have none, fetch new ones
                (None, None) => {
                    self.cur_elem = self.iter.next();
                    self.next_elem = self.iter.next();

                    // if we have the current one, set the time
                    if let Some(a) = self.cur_elem {
                        self.time = a.length;
                    }
                }
                // for the rest, we can simply exit early
                _ => return None,
            }
        }

        // and match on what to do
        match (
            self.cur_elem,
            self.cur_elem.and_then(|x| x.elem),
            self.next_elem.and_then(|x| x.elem),
        ) {
            // both elements, all are on
            (Some(a), Some(b), Some(c)) => {
                // get the blend amount
                let alpha = (self.time / a.blend_length).min(1.0);

                // and blend the 2, because alpha goes from 1 to 0, we need to blend in the other order
                Some(c.blend(b, alpha))
            }

            // only the first one, blend to silence
            (Some(a), Some(b), None) => {
                // get the blend amount
                let alpha = (self.time / a.blend_length).min(1.0);

                // and blend with a silent one
                Some(b.copy_silent().blend(b, alpha))
            }

            // only the first one, blend from silence
            (Some(a), None, Some(c)) => {
                // get the blend amount
                let alpha = (self.time / a.blend_length).min(1.0);

                // and blend with a silent one
                Some(c.blend(c.copy_silent(), alpha))
            }

            // both silent
            (Some(_), None, None) => {
                // just return silence
                Some(SynthesisElem::silent())
            }

            // nothing else, return none
            _ => None,
        }
    }
}

// and implement an easy way to get the iterator
pub trait IntoSequencer
where
    Self: IntoIterator<Item = SequenceElem> + Sized,
{
    /// creates a new sequencer, with the given sample rate
    fn sequence(self, sample_rate: u32) -> Sequencer<Self::IntoIter> {
        Sequencer {
            iter: self.into_iter(),
            delta_time: 1.0 / sample_rate as f32,
            cur_elem: None,
            next_elem: None,
            time: 0.0,
        }
    }
}

// implement it for anything that can become the right iterator
impl<T> IntoSequencer for T where T: IntoIterator<Item = SequenceElem> + Sized {}
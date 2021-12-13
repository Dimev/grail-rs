# Grail-rs (Work in progress)
Grail, A simple formant speech synthesizer, built for portability
This is the rust version

The goal of this synthesizer is to be as simple as possible, and easy to port to C and other languages if needed (I'll make a C port when this one is in a more complete state)

Still heavy WIP

# Roadmap:
 - Get the output to be roughly normalized by default
 - Complete the intonator, can see a few items into the future and adjusts voice based on that (also a ruleset for this?)
 - Complete the text->phoneme transcription, via a find-and-replace ruleset
 - Make a macro to generate a language from a language file (and do sorting automatically)
 - make a better way to make voices
 - (later) add a way to send commands to change the intonation

# License
Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

# Contribution
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
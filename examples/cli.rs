use rodio::{buffer::SamplesBuffer, OutputStream};
use std::env;

fn main() {
    // get the command line args
    let args: Vec<String> = env::args().collect();

    // figure out what to do, no args, -h or --help is print help
    // -v or --voice is to set the voice, -o or --output to set the output file path
    // -l or --langauge sets the language ruleset
    // -r or --resample to change the sample rate
    // -i or --input to read from a file
    // -s or --silent to not play back any sound
    // anything not preceeded by a switch is assumed to be speech

    // the parameters we want to use as default
    let mut voice = "";
    let mut language = "";
    let mut sample_rate = grail_rs::DEFAULT_SAMPLE_RATE;
    let mut input_file = "";
    let mut output_file = "";
    let mut play_sound = true;

    // check what we need to do
    if args.contains(&"-h".into()) || args.contains(&"--help".into()) {
        // print help menu
        // grail-rs version
        println!("Grail-rs version {}", 0);

        // flag descriptions

        // list of voices

        // list of languages

        // stop
        return;
    }

    // display them
    println!("Args: {:?}", args);

    // get an audio stream
    let (_stream, stream_handle) =
        OutputStream::try_default().expect("unable to start audio stream");

    // synthesize the speech
    let generated_audio = vec![];

    // and play it back, if needed
    if play_sound {
        stream_handle
            .play_raw(SamplesBuffer::new(
                1,
                sample_rate,
                generated_audio.clone(),
            ))
            .expect("failed to play audio");
    }
}

use grail_rs::{IntoJitter, IntoSequencer, IntoSynthesize};
use hound::{SampleFormat, WavSpec, WavWriter};
use rodio::{buffer::SamplesBuffer, OutputStream};
use std::env;
use std::fs::File;
use std::io::prelude::*;

// helps to check if there's an argument
fn has_argument(args: &[String], short: &str, long: &str) -> bool {
    // looks through the array if there is any
    args.contains(&short.into()) || args.contains(&long.into())
}

// helps find a value in a switch, if any
fn find_argument(args: &[String], short: &str, long: &str) -> Option<String> {
    // find the switch, if the first value is the right flag, the value after that is the one we need
    args.windows(2)
        .find(|x| match x {
            [switch, value] => switch.as_str() == short || switch.as_str() == long,
            _ => false,
        })
        .map(|x| x[1].clone())
}

fn main() {
    // get the command line args
    let args: Vec<String> = env::args().collect();

    // figure out what to do, no args, -h or --help is print help
    // -v or --voice is to set the voice
    // -o or --output to set the output file path
    // -l or --langauge sets the language ruleset
    // -r or --resample to change the sample rate REALLY NEEDED?
    // -i or --input to read from a file
    // -s or --silent to not play back any sound
    // -V or --version to display the version
    // anything not preceeded by a switch is assumed to be speech

    // the parameters we want to use as default
    let mut voice = String::from("sbinotto");
    let mut language = String::from("");
    let mut sample_rate = grail_rs::DEFAULT_SAMPLE_RATE;
    let mut input_file = String::new();
    let mut output_file = String::new();
    let mut play_sound = true;

    // check what we need to do
    if has_argument(&args, "-h", "--help") || args.len() < 2 {
        // print help menu
        // grail-rs version
        println!("Grail-rs version {}", 0);

        // flag descriptions

        // list of voices

        // list of languages

        // stop
        return;
    } else if has_argument(&args, "-V", "--version") {
        // print the version
        println!("Grail-rs version {}", 0);

        // stop
        return;
    }

    // now, parse the arguments with values
    if let Some(path) = find_argument(&args, "-i", "--input") {
        // open the file if it exists
        if let Ok(mut file) = File::open(path.as_str()) {
            // read the in file
            file.read_to_string(&mut input_file);
        } else {
            // give an error that we couldn't open the file
            println!("Could not open file \"{}\"", path);
            return;
        }
    }

    // set the output file, if any
    if let Some(path) = find_argument(&args, "-o", "--output") {
        output_file = path;
    }

    // do we need to be silent?
    if has_argument(&args, "-s", "--silent") {
        play_sound = false;
    }

    // what voice do we use?
    if let Some(speaker) = find_argument(&args, "-v", "--voice") {
        voice = speaker;
    }

    // figure out what to say, this is simply the last argument, if nothing is to be read from a file
    let to_say = if input_file != String::new() {
        // file was already read to here
        input_file
    } else {
        // read the last argument
        args.last().unwrap().clone()
    };

    // Display what to say
    println!("\"{}\"", to_say);
    println!(" -- {}", voice);

    // get an audio stream
    let (_stream, stream_handle) =
        OutputStream::try_default().expect("unable to start audio stream");

    // synthesize the speech
    let mut generated_audio = vec![0.0; 1024];

    // bit hacky but should work for now
    let phoneme = grail_rs::SynthesisElem::new(
        grail_rs::DEFAULT_SAMPLE_RATE,
        120.0,
        [
            810.0, 1271.0, 2851.0, 3213.0, 1.0, 1.0, 1.0, 1.0, 1200.0, 2000.0, 3000.0, 4000.0,
        ],
        [
            80.0, 120.0, 180.0, 200.0, 100.0, 100.0, 100.0, 100.0, 300.0, 120.0, 100.0, 100.0,
        ],
        [0.3, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        3000.0,
        100.0,
        0.0,
        0.1,
    );

    // put it in a sequence element
    let seq = grail_rs::SequenceElem::new(phoneme, 0.5, 0.2);

    // and extend the sound part with it
    generated_audio.extend(
        [seq]
            .sequence(grail_rs::DEFAULT_SAMPLE_RATE)
            .jitter(
                0,
                16.0 / grail_rs::DEFAULT_SAMPLE_RATE as f32,
                8.0 / grail_rs::DEFAULT_SAMPLE_RATE as f32,
                8.0 / grail_rs::DEFAULT_SAMPLE_RATE as f32,
                0.1,
            )
            .synthesize(),
    );

    // display info on how long the audio file is
    println!(
        "{} seconds of audio, generated in {} ms",
        generated_audio.len() as f32 / grail_rs::DEFAULT_SAMPLE_RATE as f32,
        1
    );

    // if there's an output file, write to it
    if output_file != String::new() {
        println!("Writing generated sound to {}", output_file);

        // this is a wav file, TODO: replace with our own code
        let spec = WavSpec {
            channels: 1,
            sample_rate: grail_rs::DEFAULT_SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(output_file.as_str(), spec).unwrap();
        for sample in generated_audio.iter() {
            writer
                .write_sample((sample * i16::MAX as f32) as i16)
                .unwrap();
        }
    }

    // and play it back, if needed // TODO put this in a function
    if play_sound {
        stream_handle
            .play_raw(SamplesBuffer::new(1, sample_rate, generated_audio.clone()))
            .expect("failed to play audio");
    }

    // wait till the sound stops playing
    if play_sound {
        std::thread::sleep(std::time::Duration::from_secs_f32(
            (generated_audio.len() as f32 / grail_rs::DEFAULT_SAMPLE_RATE as f32) + 0.1,
        ));
    }
}

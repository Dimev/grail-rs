use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use grail_rs::{
    IntoIntonator, IntoJitter, IntoSelector, IntoSequencer, IntoSynthesize, IntoTranscriber,
};

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

// save a wav file
fn save_wav(path: &str, data: &[f32], sample_rate: u32) {
    // open a file
    if let Ok(mut file) = std::fs::File::create(path) {
        // write the header
        // riff
        file.write(b"RIFF").expect("Failed to write");

        // file size, or sub chunk 2 size + 36
        file.write(&((36 + data.len() * 2) as i32).to_le_bytes())
            .expect("Failed to write");

        // wave header
        file.write(b"WAVE").expect("Failed to write");

        // format
        file.write(b"fmt ").expect("Failed to write");

        // sub chunk size
        file.write(&(16 as i32).to_le_bytes())
            .expect("Failed to write");

        // format, just 1 as we want pcm
        file.write(&(1 as i16).to_le_bytes())
            .expect("Failed to write");

        // 1 channel
        file.write(&(1 as i16).to_le_bytes())
            .expect("Failed to write");

        // sample rate
        file.write(&(sample_rate as i32).to_le_bytes())
            .expect("Failed to write");

        // byte rate, sample rate * num channels * bytes per sample
        file.write(&(sample_rate as i32 * 2).to_le_bytes())
            .expect("Failed to write");

        // block align, num channels * bytes per sample
        file.write(&(2 as i16).to_le_bytes())
            .expect("Failed to write");

        // bits per sample
        file.write(&(16 as i16).to_le_bytes())
            .expect("Failed to write");

        // data
        file.write(b"data").expect("Failed to write");

        // sub chunk size, num samples * num channels * bytes per sample
        file.write(&(data.len() as i32 * 2).to_le_bytes())
            .expect("Failed to write");

        // and write the actual sound data
        for i in data.iter().map(|x| (x * i16::MAX as f32) as i16) {
            // write the sample
            file.write(&i.to_le_bytes()).expect("Failed to write");
        }

        // and store
        file.flush().expect("Failed to write");
    }
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
        println!("Grail, a rust speech synthesizer");
        println!("The last argument is interpreted as text to be spoken");
        println!(
            "So 'grail -v bob hello' will say 'hello'. -v is to set the voice, bob in this case"
        );

        // flag descriptions
        println!("Flags:");
        println!("-v or --voice is to set the voice");
        println!("-o or --output to set the output file path");
        println!("-l or --langauge sets the language ruleset");
        println!("-r or --resample to change the sample rate");
        println!("-i or --input to read from a file");
        println!("-s or --silent to not play back any sound");
        println!("-V or --version to display the version");

        // list of voices
        println!("Voices:");

        // list of languages
        println!("Languages:");

        // stop
        return;
    } else if has_argument(&args, "-V", "--version") {
        // print the version
        println!("Grail-rs version {}", env!("CARGO_PKG_VERSION"));

        // stop
        return;
    }

    // now, parse the arguments with values
    if let Some(path) = find_argument(&args, "-i", "--input") {
        // open the file if it exists
        if let Ok(mut file) = File::open(path.as_str()) {
            // read the in file
            file.read_to_string(&mut input_file)
                .expect("Failed to read file");
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

    // synthesize the speech
    let mut generated_audio = Vec::with_capacity(sample_rate as usize * 4);

    // measure the time it takes to synthesize the audio
    let start = std::time::Instant::now();

    // and extend the sound part with it
    generated_audio.extend(
        to_say
            .chars()
            .transcribe(grail_rs::languages::generic())
            .intonate(grail_rs::languages::generic(), grail_rs::voices::generic())
            .select(grail_rs::voices::generic())
            .sequence(sample_rate)
            .jitter(0, grail_rs::voices::generic())
            .synthesize(),
    );

    let duration = start.elapsed().as_micros();

    // display info on how long the audio file is
    println!(
        "{:.2} seconds of audio, generated in {} microseconds",
        generated_audio.len() as f32 / sample_rate as f32,
        duration
    );

    // if there's an output file, write to it
    if output_file != String::new() {
        println!("Writing generated sound to {}", output_file);

        // and save the file
        save_wav(&output_file, &generated_audio, sample_rate as u32);
    }

    // and play it back, if needed
    // TODO: clean this up a bit and move CPAL to a generic func
    if play_sound {
        // get cpal's host and output device
        let host = cpal::default_host();
        let device = host.default_output_device().expect("No audio device found");

        println!("Output device: {}", device.name().unwrap());

        // get a config for the stream
        let config = device
            .supported_output_configs()
            .expect("No configs found")
            .next()
            .expect("Failed to get config")
            .with_sample_rate(cpal::SampleRate(sample_rate as u32));

        // save audio length
        let audio_len = generated_audio.len();

        // num channels
        let num_channels = config.channels() as usize;

        // consumer iterator to read the generated audio
        let mut consumer = generated_audio
            .into_iter()
            .flat_map(move |x| std::iter::repeat(x).take(num_channels));

        // make a stream to play audio with
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_output_stream(
                &config.into(),
                move |data: &mut [f32], _| {
                    for i in data {
                        *i = consumer.next().unwrap_or(0.0);
                    }
                },
                move |err| println!("Error: {:?}", err),
            ),
            cpal::SampleFormat::U16 => device.build_output_stream(
                &config.into(),
                move |data: &mut [u16], _| {
                    for i in data {
                        *i =
                            ((consumer.next().unwrap_or(0.0) * 0.5 + 0.5) * u16::MAX as f32) as u16;
                    }
                },
                move |err| println!("Error: {:?}", err),
            ),
            cpal::SampleFormat::I16 => device.build_output_stream(
                &config.into(),
                move |data: &mut [i16], _| {
                    for i in data {
                        *i = (consumer.next().unwrap_or(0.0) * i16::MAX as f32) as i16;
                    }
                },
                move |err| println!("Error: {:?}", err),
            ),
        }
        .expect("Failed to make stream");

        // play
        // can't move the expect here, as stream needs to be alive long enough
        stream.play().expect("Failed to play audio");

        // wait till the sound stops playing
        std::thread::sleep(std::time::Duration::from_secs_f32(
            (audio_len as f32 / sample_rate as f32) + 0.5,
        ));
    }
}

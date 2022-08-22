use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use grail_rs::{
    IntoIntonator, IntoJitter, IntoSelector, IntoSequencer, IntoSynthesize, IntoTranscriber,
};
use std::sync::mpsc::channel;

fn main() {
    // get cpal's host and output device
    let host = cpal::default_host();
    let device = host.default_output_device().expect("No audio device found");

    // get a config for the stream
    let config = device
        .default_output_config()
        .expect("Failed to get output config");

    println!(
        "Output device: {}, {:?}, {}",
        device.name().unwrap(),
        config.sample_rate(),
        config.channels()
    );

    // num channels
    let num_channels = config.channels() as usize;

    // make the channels
    let (sender, receiver) = channel();

    // create the audio iterator
    let mut iterator = std::iter::repeat_with(move || receiver.try_recv().unwrap_or(' '))
        .transcribe(grail_rs::languages::generic())
        .intonate(grail_rs::languages::generic(), grail_rs::voices::generic())
        .select(grail_rs::voices::generic())
        .sequence(grail_rs::voices::generic())
        .jitter(0, grail_rs::voices::generic())
        .synthesize()
        .flat_map(move |x| std::iter::repeat(x).take(num_channels));

    // make a stream to play audio with
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_output_stream(
            &config.into(),
            move |data: &mut [f32], _| {
                for i in data {
                    *i = iterator.next().unwrap_or(0.0);
                }
            },
            move |err| println!("Error: {:?}", err),
        ),
        cpal::SampleFormat::U16 => device.build_output_stream(
            &config.into(),
            move |data: &mut [u16], _| {
                for i in data {
                    *i = ((iterator.next().unwrap_or(0.0) * 0.5 + 0.5) * u16::MAX as f32) as u16;
                }
            },
            move |err| println!("Error: {:?}", err),
        ),
        cpal::SampleFormat::I16 => device.build_output_stream(
            &config.into(),
            move |data: &mut [i16], _| {
                for i in data {
                    *i = (iterator.next().unwrap_or(0.0) * i16::MAX as f32) as i16;
                }
            },
            move |err| println!("Error: {:?}", err),
        ),
    }
    .expect("Failed to make stream");

    // play
    // can't move the expect here, as stream needs to be alive long enough
    stream.play().expect("Failed to play audio");

    // read input
    for line in std::io::stdin().lines().map(|x| x.unwrap()) {
        for character in line.trim().chars().chain(Some(' ').into_iter()) {
            sender.send(character).expect("Failed to send audio");
        }
    }
}

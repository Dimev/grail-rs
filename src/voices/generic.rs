//! generic voice
use crate::voices::MKPHON;
use crate::{Voice, VoiceStorage, DEFAULT_SAMPLE_RATE};

pub fn generic() -> Voice {
    Voice {
        sample_rate: DEFAULT_SAMPLE_RATE,
        phonemes: VoiceStorage {
            silence: MKPHON([1.0; 12], [1.0; 12], [0.0; 12], 3000.0, 100.0, 0.0, 0.1),
            a: MKPHON(
                [
                    810.0, 1271.0, 2851.0, 3213.0, 1.0, 1.0, 1.0, 1.0, 1200.0, 2000.0, 3000.0,
                    4000.0,
                ],
                [
                    80.0, 120.0, 180.0, 200.0, 100.0, 100.0, 100.0, 100.0, 300.0, 120.0, 100.0,
                    100.0,
                ],
                [0.3, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                2800.0,
                500.0,
                0.6,
                0.1,
            ),
        },
        jitter_frequency: 16.0 / DEFAULT_SAMPLE_RATE as f32,
        jitter_delta_frequency: 4.0 / DEFAULT_SAMPLE_RATE as f32,
        jitter_delta_formant_frequency: 4.0 / DEFAULT_SAMPLE_RATE as f32,
        jitter_delta_amplitude: 0.1,
    }
}

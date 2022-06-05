//! generic voice
use crate::voices::MKPHON;
use crate::*;

pub fn generic() -> Voice {
    Voice {
        sample_rate: DEFAULT_SAMPLE_RATE,
        phonemes: VoiceStorage {
            a: MKPHON(
                [
                    810.0, 1271.0, 2851.0, 3213.0, 1.0, 1.0, 1.0, 1.0, 1200.0, 2000.0, 3000.0,
                    4000.0,
                ],
                [
                    80.0, 120.0, 180.0, 200.0, 100.0, 100.0, 100.0, 100.0, 300.0, 120.0, 100.0,
                    100.0,
                ],
                [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.3, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            e: MKPHON(
                [
                    910.0, 1871.0, 2851.0, 3213.0, 1.0, 1.0, 1.0, 1.0, 1200.0, 2000.0, 3000.0,
                    4000.0,
                ],
                [
                    80.0, 120.0, 180.0, 200.0, 100.0, 100.0, 100.0, 100.0, 300.0, 120.0, 100.0,
                    100.0,
                ],
                [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                [0.4, 0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1],
            ),
        },
        center_frequency: 100.0 / DEFAULT_SAMPLE_RATE as f32,
        jitter_frequency: 16.0 / DEFAULT_SAMPLE_RATE as f32,
        jitter_delta_frequency: 4.0 / DEFAULT_SAMPLE_RATE as f32,
        jitter_delta_formant_frequency: 4.0 / DEFAULT_SAMPLE_RATE as f32,
        jitter_delta_amplitude: 0.1,
    }
}

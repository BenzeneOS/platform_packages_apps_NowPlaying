use std::sync::OnceLock;

use crate::{Error, Result};

const FILTER_PHASES: usize = 4_096;
const FILTER_TAPS: usize = 35;
const FILTER_WING: usize = FILTER_PHASES * (FILTER_TAPS - 1) / 2;
const PI: f64 = f64::from_bits(0x4009_21fb_5444_2d17);
const TARGET_SAMPLE_RATE: f64 = 11_025.0;

static FILTER: OnceLock<Vec<f32>> = OnceLock::new();

fn bessel_zero(value: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half = value / 2.0;
    let mut order = 1.0;
    loop {
        let ratio = half / order;
        order += 1.0;
        term *= ratio * ratio;
        sum += term;
        if term < 1.0e-21 * sum {
            return sum;
        }
    }
}

fn make_filter() -> Vec<f32> {
    let mut filter = Vec::with_capacity(FILTER_WING);
    let beta_scale = bessel_zero(6.0).recip();
    let window_scale = (FILTER_WING - 1) as f64;
    for index in 0..FILTER_WING {
        let coefficient = if index == 0 {
            0.9
        } else {
            let angle = PI * index as f64 / FILTER_PHASES as f64;
            let sinc = (0.9 * angle).sin() / angle;
            let position = index as f64 / window_scale;
            let window = bessel_zero(6.0 * (1.0 - position * position).max(0.0).sqrt());
            sinc * window * beta_scale
        };
        filter.push(coefficient as f32);
    }
    filter
}

fn filter_wing(
    filter: &[f32],
    input: &[f32],
    mut input_index: isize,
    phase: f64,
    increment: isize,
    filter_step: f64,
) -> f32 {
    let mut offset = phase * filter_step;
    let end = FILTER_WING - usize::from(increment == 1);
    if increment == 1 && phase == 0.0 {
        offset += filter_step;
    }
    let mut value = 0.0f32;
    while (offset as usize) < end {
        let mut product = filter[offset as usize];
        product *= input[input_index as usize];
        value += product;
        offset += filter_step;
        input_index += increment;
    }
    value
}

pub(crate) fn resample(samples: &[i16], sample_rate: u32) -> Result<Vec<f32>> {
    if sample_rate == 0 {
        return Err(Error::InvalidInput(
            "PCM sample rate must be positive".into(),
        ));
    }
    let factor = TARGET_SAMPLE_RATE / f64::from(sample_rate);
    let filter_step = (factor * FILTER_PHASES as f64).min(FILTER_PHASES as f64);
    let gain = factor.min(1.0) as f32;
    let time_step = factor.recip();
    let input_offset = ((FILTER_TAPS + 1) as f64 / 2.0 * (1.0 / factor).max(1.0) + 10.0) as usize;
    let input_size = (2 * input_offset + 10).max(4_096);
    let mut input = vec![0.0f32; input_size + input_offset];
    let mut input_position = input_offset;
    let mut input_read = input_offset;
    let mut input_used = 0;
    let mut time = input_offset as f64;
    let filter = FILTER.get_or_init(make_filter);
    let mut output = Vec::with_capacity((samples.len() as f64 * factor + 2.0) as usize);

    loop {
        let copied = (input_size - input_read).min(samples.len() - input_used);
        for (destination, sample) in input[input_read..input_read + copied]
            .iter_mut()
            .zip(&samples[input_used..input_used + copied])
        {
            *destination = *sample as f32 * f32::from_bits(0x3800_0000);
        }
        input_used += copied;
        input_read += copied;

        let final_input = input_used == samples.len();
        let process_count = if final_input {
            input[input_read..input_read + input_offset].fill(0.0);
            input_read as isize - input_offset as isize
        } else {
            input_read as isize - 2 * input_offset as isize
        };
        if process_count <= 0 {
            break;
        }
        let process_count = process_count as usize;

        let end_time = time + process_count as f64;
        while time < end_time {
            let center = time as usize;
            let left_phase = time - time.floor();
            let left = filter_wing(filter, &input, center as isize, left_phase, -1, filter_step);
            let right = filter_wing(
                filter,
                &input,
                center as isize + 1,
                1.0 - left_phase,
                1,
                filter_step,
            );
            let mut value = left;
            value += right;
            value *= gain;
            output.push(value);
            time += time_step;
        }

        time -= process_count as f64;
        input_position += process_count;
        let creep = time as usize - input_offset;
        time -= creep as f64;
        input_position += creep;
        let reused = input_read - (input_position - input_offset);
        input.copy_within(input_position - input_offset..input_read, 0);
        input_read = reused;
        input_position = input_offset;
    }

    Ok(output)
}

pub(crate) fn downsample_by_two(samples: &[i16]) -> Vec<f32> {
    resample(samples, 22_050).expect("the fixed sample rate is valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_the_input_length() {
        assert_eq!(downsample_by_two(&vec![0; 176_400]).len(), 88_200);
    }

    #[test]
    fn rejects_a_zero_sample_rate() {
        assert!(resample(&[], 0).is_err());
    }

    #[test]
    fn matches_the_production_prefix() {
        let samples = [
            2307, -3446, -11841, -12952, -6592, -2734, 797, 422, 1225, -176, 2223, 2834, 1672,
            1953, 1808, 2679, 198, 1376, -6004, -10594, -12301, -17790, -20060, -21813, -21000,
            -25707, -26290, -25755, -26974, -23627, -22712, -20506, -18887, -17090, -16437, -15486,
            -12247, -10428, -7104, -5509, -8508, -8115, -9136, -11433, -10122, -9796, -9834, -8816,
            -8220, -8261, -7847, -8525, -8031, -6652, -5523, -5128, -9292, -6616, -860, -74, 2111,
            6573, 9262, 10242,
        ];
        let expected = [
            0x3cbf_ab7e,
            0xbea2_98f0,
            0xbe82_2d64,
            0x3d0e_4081,
            0xb8bc_b400,
            0x3d82_2f8c,
            0x3d82_3021,
            0x3d69_d598,
            0x3d54_b48f,
            0xbe1a_5f84,
            0xbed9_79e0,
            0xbf19_f541,
            0xbf31_35c4,
            0xbf4c_295f,
            0xbf4a_e32c,
            0xbf2f_308c,
        ]
        .map(f32::from_bits);
        assert_eq!(&downsample_by_two(&samples)[..expected.len()], expected);
    }

    #[test]
    fn matches_the_16_khz_production_fixture() {
        let samples = [
            1841, -6785, -13540, -6090, -844, 1104, 372, 1204, 2955, 1389, 2414, 1527, 1261, -4742,
            -11470, -15166, -21228, -20634, -24781, -26125, -26492, -24413, -21803, -19388, -17072,
            -16124, -13112, -9496, -5951, -7779, -8588, -10548, -10537, -9519, -9335, -7913, -8325,
            -8068, -7975, -5261, -6227, -8663, -1602, 282, 4975, 9161, 11210, 15700, 17305, 9866,
            2255, -2461, -8352, -11909, -16166, -15856, -12810, -11783, -11975, -10101, -5822,
            5188, 10899, 17550, 20694, 24339, 30349, 27561, 24246, 26618, 23568, 14376, 8842, 2438,
            5880, 8263, 716, 470, 1090, 5952, 8990, 9358, 10923, 11515, 19572, 16879, 18336, 14326,
            10242, 5588, 37, 4510, 1812, -2134, -9052, -7967,
        ];
        let expected = [
            0x3cc2_cc0e,
            0xbea2_f9e9,
            0xbe81_ec26,
            0x3d0c_f7fc,
            0x38fa_ba2b,
            0x3d82_0f29,
            0x3d82_4122,
            0x3d69_f820,
            0x3d54_9189,
            0xbe1a_5cb0,
            0xbed9_82d9,
            0xbf19_fe5d,
            0xbf31_4159,
            0xbf4c_2f13,
            0xbf4a_f2c0,
            0xbf2f_3ce0,
        ]
        .map(f32::from_bits);
        let output = resample(&samples, 16_000).unwrap();
        assert_eq!(&output[..expected.len()], expected);
        assert_eq!(resample(&vec![0; 128_000], 16_000).unwrap().len(), 88_201);
    }
}

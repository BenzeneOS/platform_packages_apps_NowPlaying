use std::f64::consts::PI;

const COS_PI_4: f32 = std::f32::consts::FRAC_1_SQRT_2;
const COS_PI_8: f32 = 0.923_879_5;
const SIN_PI_8: f32 = 0.382_683_43;

// This mirrors libsense's split-radix butterfly order because a conventional float64 FFT left 14,647 model values mismatched.
#[derive(Debug, Clone)]
pub(crate) struct GoogleFft {
    tables: Vec<Vec<f32>>,
}

impl GoogleFft {
    pub(crate) fn new() -> Self {
        let tables = [2, 4, 8, 16, 32, 64, 128]
            .into_iter()
            .map(twiddle_table)
            .collect();
        Self { tables }
    }

    pub(crate) fn process(
        &self,
        input: &[f32],
        real: &mut [f32],
        imag: &mut [f32],
        work: &mut [f32],
    ) {
        for index in 0..512 {
            work[2 * index] = input[index];
            work[2 * index + 1] = input[index + 512];
        }

        combine_c(work, self.table(128), 128);
        fft_512(&mut work[..512], self);
        fft_512_tail(&mut work[512..], self);

        real[0] = work[0];
        imag[0] = 0.0;
        real[512] = work[1];
        imag[512] = 0.0;
        for (bin, &offset) in PERMUTATION.iter().enumerate() {
            let bin = bin + 1;
            if offset < 1 {
                let offset = (-offset) as usize;
                real[bin] = work[offset];
                imag[bin] = -work[offset + 1];
            } else {
                let offset = offset as usize;
                real[bin] = work[offset];
                imag[bin] = work[offset + 1];
            }
        }
    }

    pub(crate) fn process_512(
        &self,
        input: &[f32],
        real: &mut [f32],
        imag: &mut [f32],
        work: &mut [f32],
    ) {
        for index in 0..256 {
            work[2 * index] = input[index];
            work[2 * index + 1] = input[index + 256];
        }

        combine_c(work, self.table(64), 64);
        fft_256(&mut work[..256], self);
        fft_256_tail(&mut work[256..], self);

        real[0] = work[0];
        imag[0] = 0.0;
        real[256] = work[1];
        imag[256] = 0.0;
        for (bin, &offset) in PERMUTATION[..255].iter().enumerate() {
            let bin = bin + 1;
            if offset < 1 {
                let offset = ((-offset) as usize / 2) & !1;
                real[bin] = work[offset];
                imag[bin] = -work[offset + 1];
            } else {
                let offset = (offset as usize / 2) & !1;
                real[bin] = work[offset];
                imag[bin] = work[offset + 1];
            }
        }
    }

    fn table(&self, size: usize) -> &[f32] {
        &self.tables[size.trailing_zeros() as usize - 1]
    }
}

fn twiddle_table(size: usize) -> Vec<f32> {
    (1..2 * size)
        .flat_map(|index| {
            let angle = PI * index as f64 / (4 * size) as f64;
            [angle.cos() as f32, angle.sin() as f32]
        })
        .collect()
}

fn fft_512(data: &mut [f32], fft: &GoogleFft) {
    combine_a(data, fft.table(64), 64);
    fft_256(&mut data[..256], fft);
    fft_256_tail(&mut data[256..], fft);
}

fn fft_512_tail(data: &mut [f32], fft: &GoogleFft) {
    combine_b(data, fft.table(32), 32);
    fft_128_tail(&mut data[256..384], fft);
    fft_128_tail(&mut data[384..], fft);
    fft_256_tail(&mut data[..256], fft);
}

fn fft_256(data: &mut [f32], fft: &GoogleFft) {
    combine_a(data, fft.table(32), 32);
    fft_128(&mut data[..128], fft);
    fft_128_tail(&mut data[128..], fft);
}

fn fft_256_tail(data: &mut [f32], fft: &GoogleFft) {
    combine_b(data, fft.table(16), 16);
    fft_64_tail(&mut data[128..192], fft);
    fft_64_tail(&mut data[192..], fft);
    fft_128_tail(&mut data[..128], fft);
}

fn fft_128(data: &mut [f32], fft: &GoogleFft) {
    combine_a(data, fft.table(16), 16);
    fft_64(&mut data[..64], fft);
    fft_64_tail(&mut data[64..], fft);
}

fn fft_128_tail(data: &mut [f32], fft: &GoogleFft) {
    combine_b(data, fft.table(8), 8);
    fft_32_tail(&mut data[64..96], fft);
    fft_32_tail(&mut data[96..], fft);
    fft_64_tail(&mut data[..64], fft);
}

fn fft_64(data: &mut [f32], fft: &GoogleFft) {
    combine_a(data, fft.table(8), 8);
    fft_32(&mut data[..32], fft);
    fft_32_tail(&mut data[32..], fft);
}

fn fft_64_tail(data: &mut [f32], fft: &GoogleFft) {
    combine_b(data, fft.table(4), 4);
    fft_16_tail(&mut data[32..48]);
    fft_16_tail(&mut data[48..]);
    fft_32_tail(&mut data[..32], fft);
}

fn fft_32(data: &mut [f32], fft: &GoogleFft) {
    combine_a(data, fft.table(4), 4);
    fft_16(&mut data[..16]);
    fft_16_tail(&mut data[16..]);
}

fn fft_32_tail(data: &mut [f32], fft: &GoogleFft) {
    combine_b(data, fft.table(2), 2);
    fft_8_tail(&mut data[16..24]);
    fft_8_tail(&mut data[24..]);
    fft_16_tail(&mut data[..16]);
}

fn combine_a(data: &mut [f32], table: &[f32], size: usize) {
    let half = 4 * size;
    let (left, right) = data.split_at_mut(half);
    for index in 0..2 * size {
        let pair = 2 * index;
        let left_real = left[pair];
        let left_imag = left[pair + 1];
        let right_real = right[pair];
        let right_imag = right[pair + 1];
        let difference_real = left_real - left_imag;
        let sum_real = left_real + left_imag;
        let difference_imag = right_real - right_imag;
        let sum_imag = right_real + right_imag;
        left[pair] = sum_real;
        left[pair + 1] = sum_imag;
        let (cosine, sine) = if index == 0 {
            (1.0, 0.0)
        } else {
            let table_index = 2 * (index - 1);
            (table[table_index], table[table_index + 1])
        };
        right[pair] = difference_real * cosine - difference_imag * sine;
        right[pair + 1] = difference_real * sine + difference_imag * cosine;
    }
}

fn combine_b(data: &mut [f32], table: &[f32], size: usize) {
    let quarter = 4 * size;
    let (first, rest) = data.split_at_mut(quarter);
    let (second, rest) = rest.split_at_mut(quarter);
    let (third, fourth) = rest.split_at_mut(quarter);

    for index in 0..2 * size {
        let pair = 2 * index;
        let a_real = first[pair];
        let a_imag = first[pair + 1];
        let b_real = second[pair];
        let b_imag = second[pair + 1];
        let c_real = third[pair];
        let c_imag = third[pair + 1];
        let d_real = fourth[pair];
        let d_imag = fourth[pair + 1];

        let second_imag_difference = b_imag - d_imag;
        let first_real_difference = a_real - c_real;
        let first_imag_difference = a_imag - c_imag;
        first[pair] = c_real + a_real;
        let second_real_difference = b_real - d_real;
        second[pair] = d_real + b_real;
        first[pair + 1] = a_imag + c_imag;
        second[pair + 1] = b_imag + d_imag;
        let third_real = first_real_difference - second_imag_difference;
        let third_imag = second_real_difference + first_imag_difference;
        let fourth_real = first_real_difference + second_imag_difference;
        let fourth_imag = first_imag_difference - second_real_difference;
        if index == 0 {
            third[pair] = third_real;
            third[pair + 1] = third_imag;
            fourth[pair] = fourth_real;
            fourth[pair + 1] = fourth_imag;
            continue;
        }

        if size == 2 && index == 2 {
            third[pair] = (third_real - third_imag) * COS_PI_4;
            third[pair + 1] = (third_real + third_imag) * COS_PI_4;
            fourth[pair] = (fourth_real + fourth_imag) * COS_PI_4;
            fourth[pair + 1] = (fourth_imag - fourth_real) * COS_PI_4;
            continue;
        }

        let cosine = table[pair - 2];
        let sine = table[pair - 1];
        third[pair] = third_real * cosine - third_imag * sine;
        third[pair + 1] = third_real * sine + third_imag * cosine;
        fourth[pair] = fourth_real * cosine + fourth_imag * sine;
        fourth[pair + 1] = fourth_imag * cosine - fourth_real * sine;
    }
}

fn combine_c(data: &mut [f32], table: &[f32], size: usize) {
    let half = 4 * size;
    let (left, right) = data.split_at_mut(half);
    for index in 0..2 * size {
        let pair = 2 * index;
        let left_real = left[pair];
        let left_imag = left[pair + 1];
        let right_real = right[pair];
        let right_imag = right[pair + 1];
        let left_difference = left_real - left_imag;
        let left_sum = left_real + left_imag;
        let right_difference = right_real - right_imag;
        let right_sum = right_real + right_imag;
        left[pair] = left_sum;
        left[pair + 1] = right_sum;
        let (cosine, sine) = if index == 0 {
            (1.0, 0.0)
        } else if index <= size {
            let table_index = 2 * (index - 1);
            (table[table_index], table[table_index + 1])
        } else {
            let table_index = 2 * (2 * size - index - 1);
            (table[table_index + 1], table[table_index])
        };
        if index == size {
            right[pair] = (left_difference - right_difference) * COS_PI_4;
            right[pair + 1] = (left_difference + right_difference) * COS_PI_4;
        } else {
            right[pair] = left_difference * cosine - right_difference * sine;
            right[pair + 1] = left_difference * sine + right_difference * cosine;
        }
    }
}

fn fft_8(data: &mut [f32]) {
    let a2 = data[2];
    let a3 = data[3];
    let a6 = data[6];
    let a7 = data[7];
    let a1 = data[1];
    let d23 = a2 - a3;
    let a4 = data[4];
    let a5 = data[5];
    let d67 = a6 - a7;
    let s23 = a2 + a3;
    let s67 = a6 + a7;
    let s01 = data[0] + a1;
    let d01 = data[0] - a1;
    let s45 = a4 + a5;
    let d45 = a4 - a5;
    let combined = s23 + s67;
    let rotated_difference = (d23 - d67) * COS_PI_4;
    data[2] = s01 - s45;
    data[3] = s23 - s67;
    let tail = d45 - (d23 + d67) * COS_PI_4;
    data[0] = (s01 + s45) + combined;
    data[1] = (s01 + s45) - combined;
    data[6] = d01 - rotated_difference;
    data[7] = tail;
    data[4] = d01 + rotated_difference;
    data[5] = d45 + (d23 + d67) * COS_PI_4;
}

fn fft_8_tail(data: &mut [f32]) {
    let a1 = data[1];
    let a2 = data[2];
    let a4 = data[4];
    let a5 = data[5];
    let a3 = data[3];
    let a6 = data[6];
    let a7 = data[7];
    let d15 = a1 - a5;
    let d04 = data[0] - a4;
    let s04 = a4 + data[0];
    let d26 = a2 - a6;
    let d37 = a3 - a7;
    let s26 = a6 + a2;
    let s15 = a5 + a1;
    let s37 = a7 + a3;
    let left = d15 - d26;
    let upper = d04 + d37;
    let right = d26 + d15;
    let base = s04 + s26;
    let difference = s04 - s26;
    data[6] = upper;
    data[7] = left;
    data[4] = d04 - d37;
    data[5] = right;
    data[2] = difference;
    data[3] = s15 - s37;
    data[0] = base;
    data[1] = s15 + s37;
}

fn fft_16(data: &mut [f32]) {
    let a0 = data[0];
    let a1 = data[1];
    let a8 = data[8];
    let a9 = data[9];
    let d01 = a0 - a1;
    data[0] = a0 + a1;
    data[8] = d01;
    data[1] = a8 + a9;
    data[9] = a8 - a9;

    for index in 1..4 {
        let pair = 2 * index;
        let left_real = data[pair];
        let left_imag = data[pair + 1];
        let right_real = data[pair + 8];
        let right_imag = data[pair + 9];
        let left_difference = left_real - left_imag;
        let right_difference = right_real - right_imag;
        data[pair] = left_real + left_imag;
        data[pair + 1] = right_real + right_imag;
        let (cosine, sine) = match index {
            1 => (COS_PI_8, SIN_PI_8),
            2 => (COS_PI_4, COS_PI_4),
            _ => (SIN_PI_8, COS_PI_8),
        };
        data[pair + 8] = left_difference * cosine - right_difference * sine;
        data[pair + 9] = left_difference * sine + right_difference * cosine;
    }
    fft_8(&mut data[..8]);
    fft_8_tail(&mut data[8..]);
}

fn fft_16_tail(data: &mut [f32]) {
    let mut x = [0.0; 16];
    x.copy_from_slice(data);
    let difference_08 = x[0] - x[8];
    let difference_5_13 = x[5] - x[13];
    let difference_2_10 = x[2] - x[10];
    let difference_7_15 = x[7] - x[15];
    let crossed_difference = difference_2_10 - difference_7_15;
    let crossed_sum = difference_2_10 + difference_7_15;
    let difference_3_11 = x[3] - x[11];
    let difference_6_14 = x[6] - x[14];
    let paired_sum = difference_3_11 + difference_6_14;
    let paired_difference = difference_3_11 - difference_6_14;
    let difference_4_12 = x[4] - x[12];
    let outer_difference = (x[1] - x[9]) - difference_4_12;
    let outer_sum = (x[1] - x[9]) + difference_4_12;
    let crossed_rotation = (crossed_difference + paired_sum) * COS_PI_4;
    let alternate_rotation = (crossed_difference - paired_sum) * COS_PI_4;
    let sum_rotation = (paired_difference + crossed_sum) * COS_PI_4;
    let difference_rotation = (paired_difference - crossed_sum) * COS_PI_4;
    data[0] = x[0] + x[8];
    data[1] = x[9] + x[1];
    data[2] = x[10] + x[2];
    data[3] = x[11] + x[3];
    data[4] = x[4] + x[12];
    data[5] = x[13] + x[5];
    data[6] = x[14] + x[6];
    data[7] = x[15] + x[7];
    data[8] = (difference_08 - difference_5_13) + alternate_rotation;
    data[9] = outer_sum + crossed_rotation;
    data[10] = (difference_08 - difference_5_13) - alternate_rotation;
    data[11] = outer_sum - crossed_rotation;
    data[12] = (difference_5_13 + difference_08) + sum_rotation;
    data[13] = outer_difference + difference_rotation;
    data[14] = (difference_5_13 + difference_08) - sum_rotation;
    data[15] = outer_difference - difference_rotation;
    fft_8_tail(&mut data[..8]);
}

const PERMUTATION: [i16; 511] = [
    -512, -256, 896, -128, -768, 448, 704, -64, -640, -384, 864, 224, -960, 352, 608, -32, -576,
    -320, 992, -192, -832, 432, 688, 112, -736, -480, 816, 176, -928, 304, 560, -16, -544, -288,
    944, -160, -800, 496, 752, -96, -672, -416, 856, 216, -1008, 344, 600, 56, -624, -368, 984,
    -240, -880, 408, 664, 88, -720, -464, 792, 152, -912, 280, 536, -8, -528, -272, 920, -144,
    -784, 472, 728, -80, -656, -400, 888, 248, -976, 376, 632, -48, -592, -336, 1016, -208, -848,
    428, 684, 108, -760, -504, 812, 172, -952, 300, 556, 28, -568, -312, 940, -184, -824, 492, 748,
    -120, -696, -440, 844, 204, -1000, 332, 588, 44, -616, -360, 972, -232, -872, 396, 652, 76,
    -712, -456, 780, 140, -904, 268, 524, -4, -520, -264, 908, -136, -776, 460, 716, -72, -648,
    -392, 876, 236, -968, 364, 620, -40, -584, -328, 1004, -200, -840, 444, 700, 124, -744, -488,
    828, 188, -936, 316, 572, -24, -552, -296, 956, -168, -808, 508, 764, -104, -680, -424, 854,
    214, -1020, 342, 598, 54, -636, -380, 982, -252, -892, 406, 662, 86, -732, -476, 790, 150,
    -924, 278, 534, 14, -540, -284, 918, -156, -796, 470, 726, -92, -668, -412, 886, 246, -988,
    374, 630, -60, -604, -348, 1014, -220, -860, 422, 678, 102, -756, -500, 806, 166, -948, 294,
    550, 22, -564, -308, 934, -180, -820, 486, 742, -116, -692, -436, 838, 198, -996, 326, 582, 38,
    -612, -356, 966, -228, -868, 390, 646, 70, -708, -452, 774, 134, -900, 262, 518, -2, -516,
    -260, 902, -132, -772, 454, 710, -68, -644, -388, 870, 230, -964, 358, 614, -36, -580, -324,
    998, -196, -836, 438, 694, 118, -740, -484, 822, 182, -932, 310, 566, -20, -548, -292, 950,
    -164, -804, 502, 758, -100, -676, -420, 862, 222, -1012, 350, 606, 62, -628, -372, 990, -244,
    -884, 414, 670, 94, -724, -468, 798, 158, -916, 286, 542, -12, -532, -276, 926, -148, -788,
    478, 734, -84, -660, -404, 894, 254, -980, 382, 638, -52, -596, -340, 1022, -212, -852, 426,
    682, 106, -766, -510, 810, 170, -958, 298, 554, 26, -574, -318, 938, -190, -830, 490, 746,
    -126, -702, -446, 842, 202, -1006, 330, 586, 42, -622, -366, 970, -238, -878, 394, 650, 74,
    -718, -462, 778, 138, -910, 266, 522, 6, -526, -270, 906, -142, -782, 458, 714, -78, -654,
    -398, 874, 234, -974, 362, 618, -46, -590, -334, 1002, -206, -846, 442, 698, 122, -750, -494,
    826, 186, -942, 314, 570, -30, -558, -302, 954, -174, -814, 506, 762, -110, -686, -430, 850,
    210, -1018, 338, 594, 50, -634, -378, 978, -250, -890, 402, 658, 82, -730, -474, 786, 146,
    -922, 274, 530, 10, -538, -282, 914, -154, -794, 466, 722, -90, -666, -410, 882, 242, -986,
    370, 626, -58, -602, -346, 1010, -218, -858, 418, 674, 98, -754, -498, 802, 162, -946, 290,
    546, 18, -562, -306, 930, -178, -818, 482, 738, -114, -690, -434, 834, 194, -994, 322, 578, 34,
    -610, -354, 962, -226, -866, 386, 642, 66, -706, -450, 770, 130, -898, 258, 514,
];

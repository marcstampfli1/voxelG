//! Convert a 32x32 Better Leaves tuft PNG into the 2bpp ASCII art used by
//! src/sprites.rs.
//!
//! Usage:
//!   cargo run --example convert_tuft -- path/to/oak_leaves.png
//!
//! The pack's leaf textures are few-value pixel art (oak/birch/spruce each
//! use exactly 4 distinct lumas), so tones are assigned per VALUE BUCKET by
//! the bucket's luma ratio to the DOMINANT bucket (the leaf body): the pack
//! shades relative to the body, not on an absolute scale. The thresholds are
//! the midpoints of the oak port's bucket-ratio gaps (oak buckets sit at
//! 0.66/0.77/1.00/1.23 of its body and are ported dark/mid/mid/bright, so
//! dark < 0.715 <= mid <= 1.115 < bright). Running the converter on
//! oak_leaves.png itself is the self-validation: the printed agreement
//! against the in-repo port must be >= 97% for the rule to be trusted on
//! other species (it is exactly 100%).

use voxelg::sprites;

const DIM: usize = 32;

fn main() {
    let path = std::env::args().nth(1).expect("usage: convert_tuft <32x32 RGBA png>");
    let decoder = png::Decoder::new(std::fs::File::open(&path).expect("open png"));
    let mut reader = decoder.read_info().expect("read png info");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("decode png");
    assert_eq!((info.width, info.height), (32, 32), "tuft must be 32x32");
    assert_eq!(info.color_type, png::ColorType::Rgba, "tuft must be RGBA");

    // Opaque lumas, row-major top-down (same orientation as the ASCII art).
    let mut luma = [[-1.0f32; DIM]; DIM];
    let mut opaque: Vec<f32> = Vec::new();
    for row in 0..DIM {
        for col in 0..DIM {
            let p = &buf[(row * DIM + col) * 4..(row * DIM + col) * 4 + 4];
            if p[3] >= 128 {
                let l = 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32;
                luma[row][col] = l;
                opaque.push(l);
            }
        }
    }

    // Dominant luma bucket = the leaf body (values rounded to 0.1 to merge
    // float noise within a bucket).
    let mut buckets: Vec<(i64, usize)> = Vec::new();
    for &l in &opaque {
        let key = (l * 10.0).round() as i64;
        match buckets.iter_mut().find(|(k, _)| *k == key) {
            Some((_, n)) => *n += 1,
            None => buckets.push((key, 1)),
        }
    }
    let body = buckets.iter().max_by_key(|&&(_, n)| n).unwrap().0 as f32 / 10.0;

    let tone = |row: usize, col: usize| -> u32 {
        let l = luma[row][col];
        if l < 0.0 { return 0; }
        let r = l / body;
        if r < 0.715 { 1 } else if r <= 1.115 { 2 } else { 3 }
    };
    let glyph = |t: u32| ['.', 'o', '#', '*'][t as usize];

    println!("// converted from {path}");
    for row in 0..DIM {
        let line: String = (0..DIM).map(|col| glyph(tone(row, col))).collect();
        println!("        \"{line}\",");
    }

    // Stats: coverage, centre coverage, densest 16x16 window.
    let total = opaque.len() as f32 / (DIM * DIM) as f32;
    let count_win = |x0: usize, y0: usize| -> usize {
        (y0..y0 + 16).flat_map(|r| (x0..x0 + 16).map(move |c| (r, c)))
            .filter(|&(r, c)| luma[r][c] >= 0.0)
            .count()
    };
    let centre = count_win(8, 8) as f32 / 256.0;
    let (mut best, mut best_at) = (0usize, (0usize, 0usize));
    for y0 in 0..=16 {
        for x0 in 0..=16 {
            let n = count_win(x0, y0);
            if n > best { best = n; best_at = (x0, y0); }
        }
    }
    println!("// coverage {total:.3}  centre16 {centre:.3}  densest16 {:.3} at ({}, {})",
             best as f32 / 256.0, best_at.0, best_at.1);
    let mut tf = [0usize; 4];
    for row in 0..DIM {
        for col in 0..DIM { tf[tone(row, col) as usize] += 1; }
    }
    let op = opaque.len() as f32;
    println!("// tones: body luma {body:.1}; dark {:.3} mid {:.3} bright {:.3}",
             tf[1] as f32 / op, tf[2] as f32 / op, tf[3] as f32 / op);

    // Agreement vs the in-repo oak port (meaningful when the input IS oak).
    // ASCII art row r maps to texel y = 31 - r (the encoder flips rows).
    let words = sprites::encoded();
    let mut agree = 0usize;
    for row in 0..DIM {
        for col in 0..DIM {
            if tone(row, col) == sprites::tuft_texel(&words, sprites::TUFT_OAK, col, DIM - 1 - row) {
                agree += 1;
            }
        }
    }
    println!("// agreement vs in-repo oak port: {:.1}%", 100.0 * agree as f32 / 1024.0);
}

// Hand-authored 16x16 foliage sprites, drawn as ASCII art and encoded to
// 2-bit texels for the raymarch shader (storage binding 18).
//
// This is the Allumeria/Minecraft foliage recipe: plants get their look from
// deliberately drawn cutout textures - clumped holes, silhouettes, two-tone
// shading - not from hash noise. Editing a sprite = editing the ASCII art
// below; the encoder packs it at startup and unit tests guard the dimensions,
// charset and shape contracts.
//
// Legend: '.' transparent  '#' primary  'o' secondary (dark/stem)  '*' accent
// Rows are written top-down as you read them; the encoder flips them so texel
// y = 0 is the sprite's BOTTOM row (the shader's v coordinate grows upward).
//
// Atlas layout: N_SPRITES 16x16 sprites first, then the 32x32 Better Leaves
// tufts at word offset TUFT_BASE_WORDS. All indices and offsets are defined
// ONCE here and mirrored into the shader via `wgsl_consts()` (prepended to
// the raymarch source), so the two sides can never drift.

pub const SPRITE_DIM: usize = 16;
/// 16x16 texels x 2 bits = 512 bits = 16 u32 words per sprite.
pub const SPRITE_WORDS: usize = SPRITE_DIM * SPRITE_DIM * 2 / 32;
pub const BL_TUFT_DIM: usize = 32;
/// 32x32 texels x 2 bits = 64 u32 words per tuft.
pub const BL_TUFT_WORDS: usize = BL_TUFT_DIM * BL_TUFT_DIM * 2 / 32;

macro_rules! atlas_consts {
    ($($(#[$doc:meta])* $name:ident = $val:expr;)*) => {
        $($(#[$doc])* pub const $name: usize = $val;)*

        /// WGSL mirror of every atlas constant, prepended to the raymarch
        /// shader source (like build.rs does for the world dimensions).
        pub fn wgsl_consts() -> String {
            let mut s = String::with_capacity(512);
            $(s.push_str(&format!("const {}: u32 = {}u;\n", stringify!($name), $name));)*
            s
        }
    };
}

atlas_consts! {
    /// Tuft of tapering blades, dark toward the base (classic tall grass).
    SPR_TALL_GRASS_A = 0;
    /// Wide arcing blades fanning outward.
    SPR_TALL_GRASS_B = 1;
    /// Short dense meadow clump (top rows empty - reads shorter).
    SPR_TALL_GRASS_C = 2;
    /// Sparse forked dry straw for sand/snow ground.
    SPR_DRY_TUFT = 3;
    /// Red petal head, dark centre. Flowers are contiguous from here so a
    /// species pick is `SPR_POPPY + n`.
    SPR_POPPY = 4;
    /// White radiating petals, yellow centre.
    SPR_DAISY = 5;
    /// Warm cup blossom on a stem.
    SPR_TULIP = 6;
    /// Spiky ragged head.
    SPR_CORNFLOWER = 7;
    /// Round fluffy puff head.
    SPR_DANDELION = 8;
    /// Number of 16x16 sprites in the atlas.
    N_SPRITES = 9;
    /// Word offset of the first 32x32 tuft (after the 16x16 sprites).
    TUFT_BASE_WORDS = N_SPRITES * SPRITE_WORDS;
    /// Better Leaves tuft indices: tuft i lives at word
    /// TUFT_BASE_WORDS + i * BL_TUFT_WORDS.
    TUFT_OAK = 0;
    TUFT_BIRCH = 1;
    TUFT_PINE = 2;
    /// Number of 32x32 tufts in the atlas.
    N_TUFTS = 3;
}

/// IMPORTANT (flowers): the cross-quad renderer draws the SAME sprite on two
/// diagonal planes through the voxel centre. The stem must sit exactly on the
/// centre columns (7-8, which are also mirror-invariant: 15-7 = 8) or the two
/// quads render two separate stems instead of one X.

#[rustfmt::skip]
const ART: [[&str; SPRITE_DIM]; N_SPRITES] = [
    // SPR_TALL_GRASS_A - a tuft of tapering blades, dark toward the base.
    [
        "................",
        ".....#..........",
        ".....#....#.....",
        "..#..#....#.....",
        "..#..#...##...#.",
        "..#.##...#....#.",
        "...#.#...#...##.",
        "...#.#..##...#..",
        "...#.##.#...##..",
        "....#o#.#...#...",
        ".#..#o#.#..##..#",
        ".#.#oo#o#..#..#.",
        "..#.#o#o#.##.#..",
        "..#o#oo#o##..#..",
        "...#oo#o#o#.##..",
        "..o#o#oo#oo#o...",
    ],
    // SPR_TALL_GRASS_B - wide arcing blades fanning outward from the root.
    [
        "................",
        ".#............#.",
        ".#....#...#...#.",
        "..#...#...#..#..",
        "..#...#...#..#..",
        "...#..##.##..#..",
        "...#..#o.#..##..",
        "....#.#o.#..#...",
        "....#o#..#o#....",
        ".....o#..#o.....",
        ".....o#..#o.#...",
        ".#...#o..o#..#..",
        "..#..#o..o#.#...",
        "...#o#o..o#o#...",
        "....o#o..o#o....",
        "...o#oo##oo#o...",
    ],
    // SPR_TALL_GRASS_C - short dense meadow clump; top rows stay empty so
    // the clump reads visibly shorter than A/B.
    [
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
        "....#...#..#....",
        "..#.#..##..#.#..",
        "..#.##.#o.##.#..",
        "...#.#.#o#.#.#..",
        "...#o#.#o#o#.#..",
        "..#.#o##o#o#.#..",
        "..#o#o#oo#o#o#..",
        "...#oo#o#oo#o...",
        "..o#o#oo#o#o#o..",
        ".oo#o#oo#o#oo#o.",
    ],
    // SPR_DRY_TUFT - sparse forked straw for deserts and snow: thin dark
    // stalks ('o'), a couple of lit strands ('#') at the root.
    [
        "................",
        ".o..............",
        ".o...........o..",
        "..o...o......o..",
        "..o...o.....o...",
        "...o..o.o...o...",
        "...o..o.o..o....",
        "....o.o.o..o....",
        "....o.oo.o.o....",
        ".....o.o.o.o....",
        ".....o.oo.o.....",
        "..o...oo.oo..o..",
        "...oo.o..o..o...",
        "....o.oo.o.o....",
        ".....o.ooo.o....",
        "....oo#oo#oo....",
    ],
    // SPR_POPPY - red petal head, dark centre, stem dead-centre on cols 7-8.
    [
        "................",
        "......####......",
        ".....######.....",
        ".....##**##.....",
        ".....##**##.....",
        "......####......",
        ".......##.......",
        ".......oo.......",
        ".......oo.......",
        ".......oo.......",
        "....o..oo.......",
        ".....o.oo..o....",
        "......ooo.o.....",
        ".......oo.......",
        ".......oo.......",
        ".......oo.......",
    ],
    // SPR_DAISY - white radiating petals, yellow centre, stem on cols 7-8.
    [
        "................",
        "......#..#......",
        "...#..####..#...",
        "....########....",
        "....##****##....",
        "...###****###...",
        "....##****##....",
        "....########....",
        "...#..####..#...",
        "......#..#......",
        ".......oo.......",
        ".......oo.......",
        "...o...oo...o...",
        "....o..oo..o....",
        ".....o.oo.o.....",
        ".......oo.......",
    ],
    // SPR_TULIP - closed cup blossom with pointed petal tips, stem on 7-8.
    [
        "................",
        "................",
        ".....#.##.#.....",
        ".....######.....",
        ".....#*##*#.....",
        ".....#####*.....",
        ".....######.....",
        "......####......",
        ".......oo.......",
        ".......oo.......",
        ".......oo.......",
        "....o..oo.......",
        ".....o.oo..o....",
        "......ooo.o.....",
        ".......oo.......",
        ".......oo.......",
    ],
    // SPR_CORNFLOWER - spiky ragged head with bright fringe tips, stem 7-8.
    [
        "................",
        ".......#........",
        "....#..#..#.....",
        ".....#####......",
        "...#*##*##*#....",
        "....#*###*#.....",
        "...##*###*##....",
        ".....#####......",
        "......#o#.......",
        ".......oo.......",
        ".......oo.......",
        "....o..oo.......",
        ".....o.oo..o....",
        "......ooo.o.....",
        ".......oo.......",
        ".......oo.......",
    ],
    // SPR_DANDELION - round fluffy puff head on a stem, ringed by a dark
    // seed-edge so it reads as a sphere.
    [
        "................",
        "................",
        "......o##o......",
        "....#o####o#....",
        "....o##**##o....",
        "...#o#****#o#...",
        "...o##****##o...",
        "....o##**##o....",
        "....#o####o#....",
        "......o##o......",
        ".......oo.......",
        "....o..oo.......",
        ".....o.oo..o....",
        "......ooo.o.....",
        ".......oo.......",
        ".......oo.......",
    ],
];

// ---------------------------------------------------------------------------
// Better Leaves tufts (32x32): ported from "Motschen's Better Leaves Lite"
// (github.com/TeamMidnightDust/BetterLeavesLite, MIT License, (c) Motschen) -
// the pre-rounded ragged leaf tufts its big diagonal quads carry. Converted
// from the pack's *_leaves.png: '.' = transparent, o/#/* = dark/mid/bright
// leaf pixels (the originals are grayscale and tinted in-game, exactly like
// our palette tint). The block faces sample the CENTRE 16x16 of each tuft.

#[rustfmt::skip]
const BL_TUFT_OAK: [&str; BL_TUFT_DIM] = [
        "...............#................",
        "............o.o##..##...........",
        ".........#....###*.#...*........",
        ".........#oo....#*.##...*.......",
        ".....*.*###.##..*.oo#*.*.##.....",
        "....*##*#*.*#oo..o.#*##*#*.*....",
        ".....#*.*.*#*#..##..*#*.*..#....",
        ".....*.##.***..ooo#..*.##.***...",
        "...#..*#oo.*.##.##o#..*#oo.*....",
        "..#..*#*#...oo##.###.*#*#...oo..",
        "..#..***..##.###*.#..***..##..#.",
        "......*..##oo.*#*o....*..##oo.*.",
        "...o##..####..#*...o##..####....",
        ".o..o##.###..*#ooo..o##.###..*#.",
        ".##.###*##.o*#*#.##.###*##.o*#*.",
        ".o##.*###oo.***.oo##.*###oo.**..",
        ".#.##.##*#.oo*##.####.##*#.oo*#.",
        "..#.#.*#*...ooo##.###.*#*....oo.",
        ".*.#...*.##.o.###*.#...*.##.o.#.",
        ".*.##...*#oo...*#*.##...*#oo....",
        "...o#*.*###.##..*.oo#*.*###.##..",
        "...#*##*#*.*#oo..o.#*##*#*.*#o..",
        "....*#*.*.*#*#..##..*#*.*.*#*#..",
        "..#..*.##.***..ooo#..*.##.**....",
        "...#..*#oo.*.##.##o#..*#oo.*....",
        ".....*#*#...oo##.###.*#*#.......",
        "......**..##.###*.#..***..##....",
        "......*..##oo.*#*o....*..##.....",
        "........####..#*...o##...#......",
        "........###..*.ooo..o##.........",
        ".........#..*#*..#..#...........",
        ".............*..o..#............",
];

// Converted from birch_leaves.png by examples/convert_tuft.rs (bucket-ratio
// tone rule, see that file). Birch's art has NO darker-than-body shading:
// the leaf body maps to mid with bright speckles on top - the pack's actual
// structure, airier and lighter than oak.
#[rustfmt::skip]
const BL_TUFT_BIRCH: [&str; BL_TUFT_DIM] = [
        "................#...............",
        "...............*##.**...........",
        "........*..##..**..*#.#*........",
        "......#...#.**...**...#.........",
        ".....##*.*##...#.*##.##*..#.....",
        "......**.**.#.##*.#...**.**.....",
        ".........#.**#.**..**....#.**...",
        "...*#..**..***...#.*##.**..*....",
        "....#.##*..#*...##*.#.##*...*...",
        "..*...*#.#**##.#.**...*#.#**##..",
        "...##..#..*##.*#...##..#..*##.*.",
        "..#**#..**..*.**..#**#..**..*...",
        "..**..#.*#***#..#.**..#.*#***...",
        ".**..##*.#.*#..##**..##*.#.*#...",
        "...#..**..#..**.**.#..**..#..*..",
        "...*#....##*.***..**#....##*.***",
        "..**...#..**..#.#.**...#..**....",
        ".#.**.*#*#..#..*##.**.*#*#..#...",
        "...*###**..##*.**..*###**..##*.*",
        "....#.#...#.**...**.#.#...#.**..",
        ".*##.##*.*##...#.*##.##*.*##....",
        "..#...**.**.#.##*.#...**.**.#.#.",
        "....*....#.**#.**..**....#.**...",
        "...*##.**..***...#.*##.**..***..",
        "....#.##*..#*...##*.#.##*...*...",
        "......*#.#**##.#.**...*#.#*.....",
        "....#..#..*##.*#...##..#..*#....",
        "....*....*..*.**..#**#..*.......",
        "......#.*#***#..#.**....*#......",
        ".........#..#..##**..#.*........",
        ".............*..*..#............",
        ".............*.*...*............",
];

// Converted from spruce_leaves.png by examples/convert_tuft.rs. A proper
// three-tone conifer: deep needle shadow, mid body, bright tips.
#[rustfmt::skip]
const BL_TUFT_PINE: [&str; BL_TUFT_DIM] = [
        "................................",
        "...............*..#.............",
        "..............o....#............",
        ".......*o.o*..o##.o#..#*........",
        ".......o.*..#.#.#.#*#...........",
        ".....*#o#..#.o#o.*o*o*#.#.......",
        ".....#.#.#...#*#.o.*.#.#........",
        "......o#o.#o*o*o#o#.#.o#o.#o....",
        "...#..#*#.#o#.*#.#.#..#*#.......",
        ".....*o*o#.#.##.o#o.#*o*o#.#....",
        "...#.#.*#.o#o.#.#*##o#.*#.o#....",
        "...*#.#...#*#..*o*o*#.#...#*#...",
        "...o#o.#.*o*o*#o#*.o#o.#.*o*o*..",
        "...#*#.....*.#.#.#.#*#.....*....",
        "....*o*..o..#.o#o.*o*o*..o....o.",
        "....*...#o#...#*#...*...#o#.....",
        "...o...#.#.#.*o*o*.o...#.#.#.*o.",
        "...o#.#.o#o.#..*..#o#.#.o#o.....",
        ".....#..#*#...o..#.#.#..#*#.....",
        "..o#o.#*o*o*.#o##.o#o.#*o*o*.#..",
        "..#*#..o.*..#.#.#.#*#..o.*......",
        "..o.o*#o#..#.o#o.*o*o*#o#..#....",
        "...*.#.#.#...#*#.o.*.#.#.#......",
        "....#.o#o.#o*o*o#o#.#.o#o.#.....",
        "...#..#*#.#o#.*#.#.#..#*#.#o....",
        "....#*o*o#.#.##.o#o.#*o*o..#....",
        ".......*#.o#o.#.#*##o#.*#.......",
        "......#...#*#..*o*o*#.#.........",
        "...........*o*#o#*.o#o.#........",
        "...........*.#...#.#.#..........",
        "...............#o.*o*...........",
        "................................",
];

// Tuft art table, indexed by TUFT_*.
const TUFT_ART: [&[&str; BL_TUFT_DIM]; N_TUFTS] = [&BL_TUFT_OAK, &BL_TUFT_BIRCH, &BL_TUFT_PINE];

/// Encode all sprites into the flat u32 word array the shader indexes.
/// Texel (x, y) of 16x16 sprite s lives at bit `(y*16 + x) * 2` of word block
/// `s * SPRITE_WORDS`; tuft t at bit `(y*32 + x) * 2` of word block
/// `TUFT_BASE_WORDS + t * BL_TUFT_WORDS`.
pub fn encoded() -> Vec<u32> {
    let mut out = vec![0u32; TUFT_BASE_WORDS + N_TUFTS * BL_TUFT_WORDS];
    for (si, art) in ART.iter().enumerate() {
        for (row, line) in art.iter().enumerate() {
            assert_eq!(
                line.len(),
                SPRITE_DIM,
                "sprite {si} row {row} must be {SPRITE_DIM} chars"
            );
            let y = SPRITE_DIM - 1 - row; // top-down art -> bottom-up texels
            for (x, ch) in line.bytes().enumerate() {
                let v = match ch {
                    b'.' => 0u32,
                    b'#' => 1,
                    b'o' => 2,
                    b'*' => 3,
                    _ => panic!("sprite {si} row {row}: bad char '{}'", ch as char),
                };
                let bit = (y * SPRITE_DIM + x) * 2;
                out[si * SPRITE_WORDS + bit / 32] |= v << (bit % 32);
            }
        }
    }
    for (ti, art) in TUFT_ART.iter().enumerate() {
        for (row, line) in art.iter().enumerate() {
            assert_eq!(line.len(), BL_TUFT_DIM, "tuft {ti} row {row} width");
            let y = BL_TUFT_DIM - 1 - row;
            for (x, ch) in line.bytes().enumerate() {
                let v = match ch {
                    b'.' => 0u32,
                    b'o' => 1, // NOTE: tufts use 1 = dark, 2 = mid, 3 = bright
                    b'#' => 2,
                    b'*' => 3,
                    _ => panic!("tuft {ti} row {row}: bad char '{}'", ch as char),
                };
                let bit = (y * BL_TUFT_DIM + x) * 2;
                out[TUFT_BASE_WORDS + ti * BL_TUFT_WORDS + bit / 32] |= v << (bit % 32);
            }
        }
    }
    out
}

/// Decode one 16x16 texel back out (test + tooling mirror of the WGSL
/// sprite_texel).
pub fn texel(words: &[u32], sprite: usize, x: usize, y: usize) -> u32 {
    let bit = (y * SPRITE_DIM + x) * 2;
    (words[sprite * SPRITE_WORDS + bit / 32] >> (bit % 32)) & 3
}

/// Decode one 32x32 tuft texel back out (mirror of the WGSL tuft_texel).
pub fn tuft_texel(words: &[u32], tuft: usize, x: usize, y: usize) -> u32 {
    let bit = (y * BL_TUFT_DIM + x) * 2;
    (words[TUFT_BASE_WORDS + tuft * BL_TUFT_WORDS + bit / 32] >> (bit % 32)) & 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_all_sprites() {
        let w = encoded();
        assert_eq!(w.len(), TUFT_BASE_WORDS + N_TUFTS * BL_TUFT_WORDS);
        assert_eq!(TUFT_BASE_WORDS, N_SPRITES * SPRITE_WORDS);
    }

    /// The generated WGSL consts are the shader's only source of atlas
    /// indices - guard the emission format and a couple of values.
    #[test]
    fn wgsl_consts_emitted() {
        let s = wgsl_consts();
        assert!(s.contains("const SPR_POPPY: u32 = 4u;"), "{s}");
        assert!(s.contains(&format!("const TUFT_BASE_WORDS: u32 = {TUFT_BASE_WORDS}u;")), "{s}");
        assert!(s.contains("const TUFT_PINE: u32 = 2u;"), "{s}");
    }

    /// The ported Better Leaves tufts: right size, round (transparent
    /// corners), and coverage bands measured from the source textures by
    /// examples/convert_tuft.rs (oak 0.464, birch 0.396, spruce 0.388;
    /// centre-16x16 0.672 / 0.562 / 0.625 - birch's airier centre is the
    /// pack's real structure, so its floor sits below the others).
    #[test]
    fn better_leaves_tufts_intact() {
        let w = encoded();
        for (tuft, cov_band, centre_floor) in [
            (TUFT_OAK, 0.40..=0.55, 0.60),
            (TUFT_BIRCH, 0.35..=0.45, 0.52),
            (TUFT_PINE, 0.34..=0.44, 0.58),
        ] {
            let tex = |x: usize, y: usize| tuft_texel(&w, tuft, x, y);
            for (x, y) in [(0, 0), (31, 0), (0, 31), (31, 31)] {
                assert_eq!(tex(x, y), 0, "tuft {tuft} corner ({x},{y}) must be clear");
            }
            let n: usize = (0..32)
                .flat_map(|y| (0..32).map(move |x| (x, y)))
                .filter(|&(x, y)| tex(x, y) != 0)
                .count();
            let cov = n as f32 / 1024.0;
            assert!(cov_band.contains(&cov), "tuft {tuft} coverage {cov}");
            // Centre 16x16 (used by the cube faces) must be mostly opaque.
            let nc: usize = (8..24)
                .flat_map(|y| (8..24).map(move |x| (x, y)))
                .filter(|&(x, y)| tex(x, y) != 0)
                .count();
            let cc = nc as f32 / 256.0;
            assert!(cc > centre_floor, "tuft {tuft} centre too sparse: {cc}");
        }
        // The three species must actually differ (guards against the
        // placeholder aliasing ever sneaking back).
        let differs = |a: usize, b: usize| {
            (0..32).flat_map(|y| (0..32).map(move |x| (x, y)))
                .any(|(x, y)| tuft_texel(&w, a, x, y) != tuft_texel(&w, b, x, y))
        };
        assert!(differs(TUFT_OAK, TUFT_BIRCH), "birch aliases oak");
        assert!(differs(TUFT_OAK, TUFT_PINE), "pine aliases oak");
    }

    #[test]
    fn round_trips_known_texels() {
        let w = encoded();
        // SPR_POPPY art row 3 (top-down) = ".....##**##....." -> texel y = 12.
        assert_eq!(texel(&w, SPR_POPPY, 5, 12), 1); // '#'
        assert_eq!(texel(&w, SPR_POPPY, 7, 12), 3); // '*'
        assert_eq!(texel(&w, SPR_POPPY, 0, 12), 0); // '.'
        // SPR_POPPY art row 8 ".......oo......." -> y = 7, stem at x=7.
        assert_eq!(texel(&w, SPR_POPPY, 7, 7), 2); // 'o'
    }

    /// The cross-quad renderer draws the same sprite on two planes through
    /// the voxel centre: a flower's stem must sit exactly on the centre
    /// columns 7-8 (mirror-invariant) or the X renders as two split stems.
    #[test]
    fn flower_stems_centred() {
        let w = encoded();
        for s in [SPR_POPPY, SPR_DAISY, SPR_TULIP, SPR_CORNFLOWER, SPR_DANDELION] {
            for y in [0usize, 1, 4, 5] {
                assert_eq!(texel(&w, s, 7, y), 2, "sprite {s} stem col 7 y {y}");
                assert_eq!(texel(&w, s, 8, y), 2, "sprite {s} stem col 8 y {y}");
                assert_eq!(texel(&w, s, 6, y), 0, "sprite {s} col 6 clear y {y}");
                assert_eq!(texel(&w, s, 9, y), 0, "sprite {s} col 9 clear y {y}");
            }
        }
    }

    /// Every flower needs a readable head: total opacity in a sane band and
    /// at least a couple of accent texels ('*') for the species colour
    /// table to work with.
    #[test]
    fn flower_heads_have_body_and_accents() {
        let w = encoded();
        for s in [SPR_POPPY, SPR_DAISY, SPR_TULIP, SPR_CORNFLOWER, SPR_DANDELION] {
            let n: usize = (0..SPRITE_DIM)
                .flat_map(|y| (0..SPRITE_DIM).map(move |x| (x, y)))
                .filter(|&(x, y)| texel(&w, s, x, y) != 0)
                .count();
            let o = n as f32 / 256.0;
            assert!((0.08..=0.35).contains(&o), "flower {s} opacity {o}");
            let accents = (0..SPRITE_DIM)
                .flat_map(|y| (0..SPRITE_DIM).map(move |x| (x, y)))
                .filter(|&(x, y)| texel(&w, s, x, y) == 3)
                .count();
            assert!(accents >= 2, "flower {s} needs accent texels, has {accents}");
        }
    }

    /// Every grass/straw variant must be rooted (dense base), taper toward
    /// the tips and leave the top row clear. The dry tuft is deliberately
    /// sparser, hence its lower base floor.
    #[test]
    fn grass_is_rooted_and_tapers() {
        let w = encoded();
        for (s, base_floor) in [
            (SPR_TALL_GRASS_A, 8),
            (SPR_TALL_GRASS_B, 8),
            (SPR_TALL_GRASS_C, 8),
            (SPR_DRY_TUFT, 5),
        ] {
            let row_count =
                |y: usize| (0..SPRITE_DIM).filter(|&x| texel(&w, s, x, y) != 0).count();
            assert!(row_count(0) >= base_floor, "sprite {s} base row density");
            assert!(row_count(12) <= 4, "sprite {s} tip row density");
            assert_eq!(row_count(15), 0, "sprite {s} top row clear");
        }
    }
}

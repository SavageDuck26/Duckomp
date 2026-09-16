use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use wgpu::util::DeviceExt;

// -------------------------------------------------------------------
// The compute shader (WGSL). Same algorithm as the CPU/CUDA versions:
// each GPU thread picks a candidate index `k`, reconstructs a seed
// from it via the modular inverse, and checks it against every diff.
//
// IMPORTANT: this kernel does NOT try to cover the whole candidate
// space in one dispatch. Each invocation only walks `iters_per_thread`
// steps starting from `k_start`, then stops. The Rust host code calls
// this kernel repeatedly with an increasing `k_start`, which keeps any
// single GPU dispatch short — long-running dispatches get killed by
// most GPU drivers' hang-detection watchdogs (amdgpu's default is
// 2000 ms), so batching is what makes a very long search actually
// survive to completion.
//
// The found-flag is only re-read every FLAG_CHECK_STRIDE candidates
// instead of after every single candidate, cutting most of the global
// atomic-load overhead on a single contended address.
// -------------------------------------------------------------------
const SHADER_SRC: &str = r#"
struct Params {
    target0: u64,
    range: u64,
    num_candidates: u64,
    a_inv: u64,
    c_const: u64,
    a_const: u64,
    offset_low: i64,
    offset_high: i64,
    k_start: u64,
    iters_per_thread: u64,
    num_diffs: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> diffs: array<i64>;
@group(0) @binding(2) var<storage, read_write> result_seed: array<u64>;
@group(0) @binding(3) var<storage, read_write> found_flag: array<atomic<u32>>;

const FLAG_CHECK_STRIDE: u32 = 256u;

@compute @workgroup_size(256)
fn search(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    let total_threads: u64 = u64(nwg.x) * u64(256u);
    let rng_range: u64 = u64(params.offset_high - params.offset_low) + u64(1u);

    var k: u64 = params.k_start + u64(gid.x);
    let k_limit: u64 = params.k_start + params.iters_per_thread * total_threads;

    loop {
        if (k >= params.num_candidates) {
            break;
        }
        if (k >= k_limit) {
            break;
        }
        if (atomicLoad(&found_flag[0]) != 0u) {
            return;
        }

        // Check up to FLAG_CHECK_STRIDE candidates before looking at the
        // found flag again (one global atomic load per stride instead of
        // one per candidate).
        for (var j: u32 = 0u; j < FLAG_CHECK_STRIDE; j = j + 1u) {
            if (k >= params.num_candidates) {
                break;
            }
            if (k >= k_limit) {
                break;
            }

            let state1: u64 = params.target0 + k * params.range;
            let seed: u64 = params.a_inv * (state1 - params.c_const);
            var state: u64 = seed;

            var ok = true;
            for (var i: u32 = 0u; i < params.num_diffs; i = i + 1u) {
                state = params.a_const * state + params.c_const;
                let val: i64 = params.offset_low + i64(state % rng_range);
                if (val != diffs[i]) {
                    ok = false;
                    break;
                }
            }

            if (ok) {
                let prev = atomicExchange(&found_flag[0], 1u);
                if (prev == 0u) {
                    result_seed[0] = seed;
                }
                return;
            }

            k = k + total_threads;
        }
    }
}
"#;

const A: u64 = 6364136223846793005;
const C: u64 = 1442695040888963407;

fn modinv_odd_u64(a: u64) -> u64 {
    let mut x = 1u64;
    for _ in 0..6 {
        x = x.wrapping_mul(2u64.wrapping_sub(a.wrapping_mul(x)));
    }
    x
}

// Must match the WGSL `Params` struct field-for-field (storage buffers use
// simple natural alignment, so this plain repr(C) layout lines up cleanly —
// unlike `uniform` buffers, which have stricter, easier-to-mismatch rules).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    target0: u64,
    range: u64,
    num_candidates: u64,
    a_inv: u64,
    c_const: u64,
    a_const: u64,
    offset_low: i64,
    offset_high: i64,
    k_start: u64,
    iters_per_thread: u64,
    num_diffs: u32,
    _pad: u32,
}

// Dispatch geometry.
const WORKGROUPS: u32 = 4096;
const THREADS_PER_BLOCK: u32 = 256;

// amdgpu's default GPU scheduler timeout is 2000 ms. We target ~0.35 s per
// dispatch (~5.7x margin) so thermal throttling or driver hiccups can't push
// a batch past the watchdog.
const TARGET_BATCH_SECS: f64 = 0.35;

// Calibration probe: a small dispatch sized to finish in ~0.3-0.6 s on most
// discrete GPUs, run a few times to get a conservative throughput estimate.
const PROBE_ITERS_PER_THREAD: u64 = 2000;
const PROBE_RUNS: u32 = 3;

/// Runs one compute dispatch with the given params and reads back the
/// (found_flag, result_seed) pair. Blocks until the GPU finishes.
#[allow(clippy::too_many_arguments)]
fn run_batch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups: u32,
    params_buf: &wgpu::Buffer,
    result_seed_buf: &wgpu::Buffer,
    found_flag_buf: &wgpu::Buffer,
    readback_seed: &wgpu::Buffer,
    readback_flag: &wgpu::Buffer,
    params: &Params,
) -> (u32, u64) {
    queue.write_buffer(params_buf, 0, bytemuck::bytes_of(params));

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(result_seed_buf, 0, readback_seed, 0, 8);
    encoder.copy_buffer_to_buffer(found_flag_buf, 0, readback_flag, 0, 4);
    queue.submit(Some(encoder.finish()));

    let seed_slice = readback_seed.slice(..);
    let flag_slice = readback_flag.slice(..);
    seed_slice.map_async(wgpu::MapMode::Read, |_| {});
    flag_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);

    let found: u32 = bytemuck::cast_slice(&flag_slice.get_mapped_range())[0];
    let seed: u64 = bytemuck::cast_slice(&seed_slice.get_mapped_range())[0];
    readback_seed.unmap();
    readback_flag.unmap();

    (found, seed)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "sample.txt";
    let file = File::open(path).expect("Failed to open sample.txt");
    let reader = BufReader::new(file);

    let mut values: Vec<i64> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        values.push(trimmed.parse()?);
    }

    if values.len() < 2 {
        println!("Need at least 2 values in sample.txt to compute diffs.");
        return Ok(());
    }

    let diffs: Vec<i64> = values.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    let offset_low = *diffs.iter().min().unwrap();
    let offset_high = *diffs.iter().max().unwrap();
    let range = (offset_high - offset_low + 1) as u64;

    println!("Target diffs: {:?}", diffs);
    println!("Generator range: [{}, {}] (size {})", offset_low, offset_high, range);

    let a_inv = modinv_odd_u64(A);
    let target0 = (diffs[0] - offset_low) as u64;
    let num_candidates = (u64::MAX / range).saturating_add(1);

    println!("Candidates to check: ~{}", num_candidates);

    pollster::block_on(run_gpu(
        &values,
        &diffs,
        offset_low,
        offset_high,
        range,
        target0,
        a_inv,
        num_candidates,
    ))
}

async fn run_gpu(
    values: &[i64],
    diffs: &[i64],
    offset_low: i64,
    offset_high: i64,
    range: u64,
    target0: u64,
    a_inv: u64,
    num_candidates: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("No suitable GPU adapter found. Check your Vulkan drivers.");

    println!("Using adapter: {:?}", adapter.get_info());

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::SHADER_INT64,
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
            None,
        )
        .await?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("search_shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    let workgroups: u32 = WORKGROUPS;
    let threads_per_block: u32 = THREADS_PER_BLOCK;
    let total_threads: u64 = (workgroups as u64) * (threads_per_block as u64);

    let diffs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("diffs"),
        contents: bytemuck::cast_slice(diffs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let result_seed_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("result_seed"),
        size: 8,
        // COPY_DST so the host can reset it after calibration.
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let found_flag_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("found_flag"),
        contents: bytemuck::cast_slice(&[0u32]),
        // COPY_DST so the host can reset it after calibration.
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    // Params buffer is rewritten (via queue.write_buffer) with a new
    // k_start before every dispatch, so it needs COPY_DST too.
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: std::mem::size_of::<Params>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let readback_seed = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback_seed"),
        size: 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let readback_flag = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback_flag"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: diffs_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: result_seed_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: found_flag_buf.as_entire_binding() },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "search",
        compilation_options: Default::default(),
        cache: None,
    });

    // ------------------------------------------------------------------
    // Calibration: measure real per-dispatch throughput on THIS GPU/driver
    // so the batch size can be sized to stay well under the driver
    // watchdog timeout (2000 ms on amdgpu), no matter how expensive the
    // emulated u64 modulo turns out to be. We take the slowest of several
    // probe runs so later batches still have margin under throttling.
    // ------------------------------------------------------------------
    let probe_params = Params {
        target0,
        range,
        num_candidates,
        a_inv,
        c_const: C,
        a_const: A,
        offset_low,
        offset_high,
        k_start: 0,
        iters_per_thread: PROBE_ITERS_PER_THREAD,
        num_diffs: diffs.len() as u32,
        _pad: 0,
    };

    let probe_checked = total_threads * PROBE_ITERS_PER_THREAD;

    println!(
        "Calibrating: {} probe dispatches of {} candidates each...",
        PROBE_RUNS, probe_checked
    );

    let mut probe_slowest: f64 = 0.0;
    for i in 0..PROBE_RUNS {
        let t = Instant::now();
        let (found, seed) = run_batch(
            &device,
            &queue,
            &pipeline,
            &bind_group,
            workgroups,
            &params_buf,
            &result_seed_buf,
            &found_flag_buf,
            &readback_seed,
            &readback_flag,
            &probe_params,
        );
        let elapsed = t.elapsed().as_secs_f64();
        probe_slowest = probe_slowest.max(elapsed);

        if found == 1 {
            println!(
                "Calibration run {} found seed {} (main loop will rediscover it)",
                i + 1, seed
            );
        }
        println!(
            "Calibration run {}: {} candidates in {:.3}s ({:.0} candidates/s)",
            i + 1,
            probe_checked,
            elapsed,
            probe_checked as f64 / elapsed
        );
    }

    let candidates_per_sec = probe_checked as f64 / probe_slowest;
    let iters_per_thread: u64 = ((TARGET_BATCH_SECS * candidates_per_sec)
        / total_threads as f64)
        .floor()
        .max(1.0) as u64;
    let batch_size = total_threads * iters_per_thread;

    println!(
        "Calibrated: ~{:.0} candidates/s -> {} iters/thread -> {} candidates/batch ({:.3}s/batch)",
        candidates_per_sec,
        iters_per_thread,
        batch_size,
        batch_size as f64 / candidates_per_sec
    );

    // Calibration may have set the flag/seed; reset before the real search.
    queue.write_buffer(&found_flag_buf, 0, bytemuck::cast_slice(&[0u32]));
    queue.write_buffer(&result_seed_buf, 0, bytemuck::cast_slice(&[0u64]));

    let start_time = Instant::now();
    let mut k_start: u64 = 0;
    let mut found_result: Option<u64> = None;
    let mut batch_num: u64 = 0;

    while k_start < num_candidates {
        let params = Params {
            target0,
            range,
            num_candidates,
            a_inv,
            c_const: C,
            a_const: A,
            offset_low,
            offset_high,
            k_start,
            iters_per_thread,
            num_diffs: diffs.len() as u32,
            _pad: 0,
        };
        let (found, seed) = run_batch(
            &device,
            &queue,
            &pipeline,
            &bind_group,
            workgroups,
            &params_buf,
            &result_seed_buf,
            &found_flag_buf,
            &readback_seed,
            &readback_flag,
            &params,
        );

        batch_num += 1;
        k_start = k_start.saturating_add(batch_size);

        if found == 1 {
            found_result = Some(seed);
            break;
        }

        // Progress update every batch. Expect roughly one line per
        // TARGET_BATCH_SECS.
        let checked = k_start.min(num_candidates);
        let percent = 100.0 * (checked as f64) / (num_candidates as f64);
        let elapsed = start_time.elapsed().as_secs_f64();
        let rate = (checked as f64) / elapsed.max(0.001);
        println!(
            "Batch {}: {:.8}% ({} / {}) | {:.0} candidates/sec | {:.1}s elapsed",
            batch_num, percent, checked, num_candidates, rate, elapsed
        );
    }

    match found_result {
        Some(seed) => {
            println!("\n=== FOUND SEED: {} ===", seed);

            let mut state = seed;
            let mut reconstructed = vec![values[0]];
            for _ in 0..diffs.len() {
                state = A.wrapping_mul(state).wrapping_add(C);
                let step = offset_low + (state % range) as i64;
                reconstructed.push(*reconstructed.last().unwrap() + step);
            }
            println!("Reconstructed: {:?}", reconstructed);
            println!("Original:      {:?}", values);
            println!("Match: {}", reconstructed == values);
        }
        None => {
            println!("\nNo match found after exhausting the full candidate space.");
        }
    }

    Ok(())
}
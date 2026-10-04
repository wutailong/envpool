# Focused native conversion loop

These three sources preserve the measured algorithm. Publication normalization
adds license/formatting, a repository-style header guard and explicit integer
casts in clock helpers. A fresh retained-header build passes 16 tiny positive
checks: shapes 1, 20, 3×5 and 2×3×4×5, conversion/calibration, `(iterations,warmup)`
of `(1,0)` and `(17,3)`. Their timings are discarded. Reported performance comes
from the original frozen probe, not a second favorable timing campaign.

## Build boundary

This is a focused Linux extension, not an environment client or portable clean
dependency installer. Reuse matching external dependencies from a successful
release build, with the exact selected source revision in its own source root.
The observed build uses GCC 14.2, C++17, Python 3.12 headers, pybind11 3.1.0,
repository-pinned Abseil headers/PIC archives, concurrentqueue, threadpool,
OpenXLA FFI headers and MuJoCo headers. The full environment build recipe is
described in [published source rebuild](../../published_rebuild/README.md).

Compile the consumer and probe as **separate translation units** and disable
LTO in both compilation and linking. The measured command uses these flags:

```sh
CXXFLAGS='-std=c++17 -O3 -g0 -DNDEBUG -fPIC -fvisibility=hidden
 -fno-omit-frame-pointer -ffunction-sections -fdata-sections
 -fstack-protector -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=1 -pthread -fno-lto'
```

Supply the following parameters from the selected release build; they are
placeholders, not supplied caches or inferred paths:

- `SOURCE`: immutable checkout/header snapshot to measure.
- `TOOLS`: this directory in the report checkout.
- `OUT`: a new, empty output directory, different for every source variant.
- `INCLUDES`: shell array of `-I` and dependency-directory pairs, including the
  selected Python/pybind headers. Put `-I "$SOURCE"` before dependency roots.
- `ABSL_PIC_ARCHIVES`: shell array of Abseil `.pic.a` archives in the release
  link order. Other environment/renderer archives are unnecessary.

```sh
for unit in conversion_timing_probe conversion_timing_consumer; do
  g++ $CXXFLAGS -I "$SOURCE" "${INCLUDES[@]}" \
    '-DCONVERSION_TIMING_VARIANT="selected"' \
    -MD -MF "$OUT/$unit.d" -c "$TOOLS/$unit.cc" -o "$OUT/$unit.o"
done
g++ "$OUT/conversion_timing_probe.o" "$OUT/conversion_timing_consumer.o" \
  -shared -o "$OUT/envpool_conversion_timing.so" -B/usr/bin \
  -Wl,-no-as-needed -Wl,-z,relro,-z,now -pass-exit-codes -Wl,--gc-sections \
  "${ABSL_PIC_ARCHIVES[@]}" -Wl,-Bsymbolic -ldl -lrt -pthread \
  -Wl,--push-state,-as-needed -lstdc++ -Wl,--pop-state \
  -Wl,--push-state,-as-needed -lm -Wl,--pop-state -fno-lto
```

The local verification compiles these public sources with the same resolved
include/archive lists as the original measured retained build. Before measuring,
inspect `.d` files to exclude headers from a different EnvPool checkout; record
source/compiler/dependency/archive/binary SHA-256s. Inspect disassembly for the
separate consumer call and per-iteration owner release. Do not use replacement
allocators, `-Bsymbolic-functions`, sanitizers or a preloaded allocator here.
`-Bsymbolic` is retained from the original release-link template.

## Ordinary positive call

Use a fresh process for each binary; verify its SHA-256 before importing and
never replace a loaded file. The module name must remain
`envpool_conversion_timing`. Load it by exact path with `importlib.util`, create
`np.arange(1, 21, dtype=np.int32)`, and call `module.measure(input, 17, 3)` first.
The result must have equal actual/expected warmup and measured checksums and
`input_reference_count_balanced=True`. No timing claim follows from that smoke.

The fixed study changes that call to `measure(input, 1_000_000, 10_000)`;
`calibrate_consumer` uses the same arguments on one preconstructed Array.
Inputs, Python dispatch, validation and result construction are outside native
clocks. The GIL stays held by the calling thread. Default placement and
`taskset -c 0` are separate conditions, not pooled. Use the complete balanced
block schedule and A/A comparisons in the parent report; do not subtract the
consumer calibration or extrapolate to background-thread releases.

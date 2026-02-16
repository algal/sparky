/*
 * apm_shim.cpp — C-ABI wrapper around WebRTC AudioProcessing (full AEC3).
 *
 * Exposes a minimal set of functions for Python ctypes to drive the APM:
 *   apm_create()           → create & configure APM
 *   apm_process_reverse()  → feed far-end (speaker) reference
 *   apm_process()          → process near-end (mic) in-place
 *   apm_destroy()          → cleanup
 *
 * Build:
 *   g++ -shared -fPIC -o libwebrtc_apm_shim.so apm_shim.cpp \
 *       $(pkg-config --cflags --libs webrtc-audio-processing-1)
 */

#include <cstdint>
#include <cstring>
#include <modules/audio_processing/include/audio_processing.h>

// The APM instance wrapper — holds the APM pointer and the StreamConfig
// so we don't need to reconstruct them on every call.
struct ApmHandle {
    webrtc::AudioProcessing* apm;
    webrtc::StreamConfig stream_config;

    ApmHandle(webrtc::AudioProcessing* a, int sample_rate, int num_channels)
        : apm(a),
          stream_config(sample_rate, static_cast<size_t>(num_channels)) {}
};

extern "C" {

/*
 * Create and configure an APM instance.
 *
 * sample_rate:   8000, 16000, 32000, or 48000
 * num_channels:  typically 1 (mono)
 *
 * Returns an opaque handle, or NULL on failure.
 */
void* apm_create(int sample_rate, int num_channels) {
    // Create APM via the builder (returns a raw pointer with refcount).
    webrtc::AudioProcessingBuilder builder;
    webrtc::AudioProcessing* apm = builder.Create();
    if (!apm) {
        return nullptr;
    }

    // Configure: enable AEC3 (full, not mobile) and noise suppression.
    webrtc::AudioProcessing::Config config;

    // Echo canceller: full AEC3 (not mobile mode).
    config.echo_canceller.enabled = true;
    config.echo_canceller.mobile_mode = false;

    // Noise suppression at moderate level (helps AEC performance).
    config.noise_suppression.enabled = true;
    config.noise_suppression.level =
        webrtc::AudioProcessing::Config::NoiseSuppression::kModerate;

    // High-pass filter to remove DC offset (recommended for AEC).
    config.high_pass_filter.enabled = true;

    // Disable AGC — we don't want gain changes, just echo cancellation.
    config.gain_controller1.enabled = false;
    config.gain_controller2.enabled = false;

    apm->ApplyConfig(config);

    // Initialize with matching rates for capture and render.
    int rc = apm->Initialize(
        sample_rate,   // capture input rate
        sample_rate,   // capture output rate
        sample_rate,   // render (reverse) rate
        webrtc::AudioProcessing::kMono,  // capture input layout
        webrtc::AudioProcessing::kMono,  // capture output layout
        webrtc::AudioProcessing::kMono   // render layout
    );

    if (rc != 0) {
        // Release the APM on init failure.
        apm->Release();
        return nullptr;
    }

    ApmHandle* handle = new ApmHandle(apm, sample_rate, num_channels);
    return static_cast<void*>(handle);
}

/*
 * Feed far-end (speaker/render) audio into the APM.
 *
 * This must be called BEFORE apm_process() for each time-aligned frame.
 * The APM uses this to learn the echo path.
 *
 * handle:       opaque handle from apm_create()
 * far_end_data: pointer to interleaved int16 PCM samples
 * num_samples:  number of samples (must be 10ms worth, e.g. 160 for 16kHz)
 *
 * Returns 0 on success, non-zero on error.
 */
int apm_process_reverse(void* handle, const int16_t* far_end_data,
                        int num_samples) {
    if (!handle || !far_end_data) return -1;

    ApmHandle* h = static_cast<ApmHandle*>(handle);

    // ProcessReverseStream with int16 — src and dest can be the same,
    // but we don't need the output, so we use a temporary buffer.
    // Actually the API allows src==dest or a separate dest.
    // We'll use a stack buffer for the output (which we discard).
    int16_t discard[480];  // max 10ms at 48kHz
    if (num_samples > 480) return -2;

    return h->apm->ProcessReverseStream(
        far_end_data,
        h->stream_config,
        h->stream_config,
        discard
    );
}

/*
 * Process near-end (mic/capture) audio through the APM, in-place.
 *
 * handle:           opaque handle from apm_create()
 * near_end_data:    pointer to interleaved int16 PCM samples (modified in-place)
 * num_samples:      number of samples (must be 10ms worth)
 *
 * Returns 0 on success, non-zero on error.
 */
int apm_process(void* handle, int16_t* near_end_data, int num_samples) {
    if (!handle || !near_end_data) return -1;

    ApmHandle* h = static_cast<ApmHandle*>(handle);

    // Set stream delay to 0 — the caller is responsible for time-alignment
    // of the far-end and near-end streams. A small delay estimate can help
    // but 0 is a safe default when we feed frames synchronously.
    h->apm->set_stream_delay_ms(0);

    // ProcessStream with int16 — src and dest can be the same buffer.
    return h->apm->ProcessStream(
        near_end_data,
        h->stream_config,
        h->stream_config,
        near_end_data
    );
}

/*
 * Reinitialize the APM (reset internal state without destroying it).
 * Useful for clear() operations.
 *
 * Returns 0 on success.
 */
int apm_reinitialize(void* handle) {
    if (!handle) return -1;
    ApmHandle* h = static_cast<ApmHandle*>(handle);
    return h->apm->Initialize();
}

/*
 * Destroy the APM instance and free resources.
 */
void apm_destroy(void* handle) {
    if (!handle) return;

    ApmHandle* h = static_cast<ApmHandle*>(handle);
    if (h->apm) {
        h->apm->Release();
        h->apm = nullptr;
    }
    delete h;
}

}  // extern "C"

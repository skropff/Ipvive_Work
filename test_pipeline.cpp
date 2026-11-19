#include "pipeline.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <new>
#include <thread>
#include <vector>
#include <optional>

// --- Heap Allocation Trap ---
static std::atomic<bool> g_forbid_heap{false};

void* operator new(std::size_t size) {
    if (g_forbid_heap.load(std::memory_order_relaxed)) {
        std::cerr << "\n[FAIL] Heap allocation detected in hot path!\n";
        std::abort();
    }
    return std::malloc(size);
}

void operator delete(void* ptr) noexcept { std::free(ptr); }
void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }

// --- Mock Hardware State ---
struct MockHardwareState {
    std::atomic<int> projection_calls{0};
    std::atomic<int> dma_submits{0};
    const ManifoldFrame* last_submitted_ptr{nullptr};
    uint64_t last_submitted_ts{0};
    float last_submitted_coord0{0.0f};
    std::vector<const ManifoldFrame*> submitted_history;
    bool auto_complete_npu_irq{false};
    SensorNpuPipeline<4>* active_pipeline{nullptr};

    void reset(SensorNpuPipeline<4>* pipe, bool auto_irq) {
        projection_calls = 0;
        dma_submits = 0;
        last_submitted_ptr = nullptr;
        last_submitted_ts = 0;
        last_submitted_coord0 = 0.0f;
        submitted_history.clear();
        auto_complete_npu_irq = auto_irq;
        active_pipeline = pipe;
    }
} g_hw;

extern "C" {
    void project_spi_to_hyperbolic(const uint8_t* raw_spi_dma, ManifoldFrame* out_frame) {
        g_hw.projection_calls.fetch_add(1, std::memory_order_relaxed);
        out_frame->coordinates[0] = static_cast<float>(raw_spi_dma[0]);
    }

    void npu_submit_dma_descriptor(const ManifoldFrame* frame_ptr) {
        g_hw.last_submitted_ptr = frame_ptr;
        g_hw.last_submitted_ts = frame_ptr->timestamp_ns;
        g_hw.last_submitted_coord0 = frame_ptr->coordinates[0];
        if (!g_hw.auto_complete_npu_irq) {
            g_hw.submitted_history.push_back(frame_ptr);
        }
        g_hw.dma_submits.fetch_add(1, std::memory_order_release);

        if (g_hw.auto_complete_npu_irq && g_hw.active_pipeline) {
            g_hw.active_pipeline->on_npu_dma_complete_isr();
        }
    }
}

// --- Test Case 1: Compile-Time Alignment & Size ---
void test_memory_alignment() {
    static_assert(alignof(ManifoldFrame) == 64, "ManifoldFrame must be 64-byte aligned");
    static_assert(sizeof(ManifoldFrame) % 64 == 0, "ManifoldFrame size must be a multiple of 64 bytes");
    std::cout << "[PASS] Test 1: 64-byte DMA alignment verified.\n";
}

// --- Test Case 2: Basic Single-Frame Zero-Copy Flow ---
void test_single_frame_handoff() {
    SensorNpuPipeline<4> pipe;
    g_hw.reset(&pipe, false);

    uint8_t dummy_spi[4] = {42, 0, 0, 0};
    printf("A\n");
    pipe.on_spi_dma_complete_isr(dummy_spi, 1000);
    printf("B\n");
    assert(g_hw.projection_calls == 1);

    pipe.process_and_dispatch_to_npu();
    printf("C\n");
    assert(g_hw.dma_submits == 1);
    assert(g_hw.last_submitted_ts == 1000);
    assert(g_hw.last_submitted_coord0 == 42.0f);
    assert(reinterpret_cast<uintptr_t>(g_hw.last_submitted_ptr) % 64 == 0);

    pipe.on_npu_dma_complete_isr();
    std::cout << "[PASS] Test 2: Single-frame zero-copy handoff verified.\n";
}

// --- Test Case 3: Buffer Full / NPU Stall Backpressure ---
void test_npu_stall_backpressure() {
    SensorNpuPipeline<4> pipe;
    g_hw.reset(&pipe, false);
    uint8_t dummy_spi[4] = {10, 0, 0, 0};

    // Fill all 4 slots without NPU completing
    for (uint64_t i = 1; i <= 4; ++i) {
        dummy_spi[0] = static_cast<uint8_t>(i);
        pipe.on_spi_dma_complete_isr(dummy_spi, i * 100);
    }
    assert(g_hw.projection_calls == 4);

    // 5th and 6th interrupts must NOT block when buffer is full
    dummy_spi[0] = 99;
    pipe.on_spi_dma_complete_isr(dummy_spi, 500);
    pipe.on_spi_dma_complete_isr(dummy_spi, 600);

    // Drain 1 frame to NPU and complete NPU DMA
    pipe.process_and_dispatch_to_npu();
    pipe.on_npu_dma_complete_isr();

    // Now 1 slot is free again; next ISR must succeed
    dummy_spi[0] = 77;
    pipe.on_spi_dma_complete_isr(dummy_spi, 700);

    // Drain remaining 4 frames
    for (int i = 0; i < 4; ++i) {
        pipe.process_and_dispatch_to_npu();
        pipe.on_npu_dma_complete_isr();
    }
    assert(g_hw.dma_submits == 5);
    std::cout << "[PASS] Test 3: Non-blocking backpressure and slot recovery verified.\n";
}

// --- Test Case 4: Ring Buffer Wrap-Around Address Reuse ---
void test_ring_wraparound() {
    SensorNpuPipeline<4> pipe;
    g_hw.reset(&pipe, false);
    uint8_t dummy_spi[4] = {1, 0, 0, 0};

    for (uint64_t i = 0; i < 12; ++i) {
        pipe.on_spi_dma_complete_isr(dummy_spi, i);
        pipe.process_and_dispatch_to_npu();
        pipe.on_npu_dma_complete_isr();
    }

    assert(g_hw.submitted_history.size() == 12);
    // Slots 0..3 must repeat identically at indices 4..7 and 8..11
    for (size_t i = 0; i < 4; ++i) {
        assert(g_hw.submitted_history[i] == g_hw.submitted_history[i + 4]);
        assert(g_hw.submitted_history[i] == g_hw.submitted_history[i + 8]);
    }
    std::cout << "[PASS] Test 4: Ring buffer address wrap-around verified.\n";
}

// --- Test Case 5: Multithreaded Stress + Zero Heap Allocation ---
void test_concurrent_stress_and_no_heap() {
    SensorNpuPipeline<4> pipe;
    g_hw.reset(&pipe, true); // Auto-trigger NPU completion ISR on submit

    constexpr int kTotalFrames = 50000;
    std::atomic<bool> stop_worker{false};

    g_forbid_heap.store(true, std::memory_order_release);

    std::thread worker([&]() {
        while (!stop_worker.load(std::memory_order_acquire)) {
            pipe.process_and_dispatch_to_npu();
        }
    });

    std::thread spi_isr([&]() {
        uint8_t data[4] = {1, 2, 3, 4};
        for (int i = 1; i <= kTotalFrames; ++i) {
            data[0] = static_cast<uint8_t>(i & 0xFF);
            pipe.on_spi_dma_complete_isr(data, static_cast<uint64_t>(i));
        }
        // Wait briefly for worker to drain in-flight frames
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        stop_worker.store(true, std::memory_order_release);
        // Send 1 wake-up frame so worker exits cleanly
        pipe.on_spi_dma_complete_isr(data, 999999);
    });

    spi_isr.join();
    worker.join();

    g_forbid_heap.store(false, std::memory_order_release);

    assert(g_hw.dma_submits > 0);
    std::cout << "[PASS] Test 5: Concurrent stress (" << g_hw.dma_submits
              << " frames) with zero heap allocations verified.\n";
}

int main() {
    test_memory_alignment();
    test_single_frame_handoff();
    test_npu_stall_backpressure();
    test_ring_wraparound();
    test_concurrent_stress_and_no_heap();
    std::cout << "\nALL TESTS PASSED.\n";
    return 0;
}

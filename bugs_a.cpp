#include "pipeline.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

struct MockHardware {
    std::atomic<int> projection_calls{0};
    std::atomic<int> dma_submits{0};
    std::vector<const ManifoldFrame*> submitted_ptrs;
    std::vector<uint64_t> submitted_timestamps;
    std::atomic<bool> in_flight_overwrite{false};
    int slot_in_flight[4]{0, 0, 0, 0};
    const ManifoldFrame* pool_base{nullptr};

    void reset(const ManifoldFrame* base) {
        projection_calls = 0;
        dma_submits = 0;
        submitted_ptrs.clear();
        submitted_timestamps.clear();
        in_flight_overwrite = false;
        pool_base = base;
        for (int i = 0; i < 4; ++i) slot_in_flight[i] = 0;
    }
} g_mock;

extern "C" {
    void project_spi_to_hyperbolic(const uint8_t* raw_spi_dma, ManifoldFrame* out_frame) {
        g_mock.projection_calls.fetch_add(1, std::memory_order_relaxed);
        if (g_mock.pool_base != nullptr) {
            ptrdiff_t idx = out_frame - g_mock.pool_base;
            if (idx >= 0 && idx < 4 && g_mock.slot_in_flight[idx] > 0) {
                g_mock.in_flight_overwrite.store(true, std::memory_order_release);
            }
        }
        out_frame->coordinates[0] = static_cast<float>(raw_spi_dma[0]);
    }

    void npu_submit_dma_descriptor(const ManifoldFrame* frame_ptr) {
        if (g_mock.pool_base == nullptr) {
            g_mock.pool_base = frame_ptr; // First frame is always slot 0
        }
        ptrdiff_t idx = frame_ptr - g_mock.pool_base;
        if (idx >= 0 && idx < 4) {
            g_mock.slot_in_flight[idx]++;
        }
        g_mock.submitted_ptrs.push_back(frame_ptr);
        g_mock.submitted_timestamps.push_back(frame_ptr->timestamp_ns);
        g_mock.dma_submits.fetch_add(1, std::memory_order_release);
    }
}

// Test 1: Multiple frames in flight on NPU when a single NPU completion IRQ fires
void test_single_npu_completion_per_frame() {
    SensorNpuPipeline<4> pipe;
    g_mock.reset(nullptr);
    uint8_t spi[4] = {10, 0, 0, 0};

    // 1. Fill all 4 slots
    for (uint64_t i = 1; i <= 4; ++i) {
        spi[0] = static_cast<uint8_t>(i);
        pipe.on_spi_dma_complete_isr(spi, i * 100);
    }

    // 2. Dispatch all 4 frames to NPU (all 4 are now simultaneously in flight on NPU)
    for (int i = 0; i < 4; ++i) {
        pipe.process_and_dispatch_to_npu();
    }
    assert(g_mock.dma_submits == 4);

    // 3. NPU finishes ONLY ONE frame (slot 0) and fires a single completion ISR.
    // Slots 1, 2, and 3 are STILL being read by the NPU!
    pipe.on_npu_dma_complete_isr();
    g_mock.slot_in_flight[0]--;

    // 4. Two new SPI interrupts arrive back-to-back:
    //    - The 1st should take the 1 newly freed slot (slot 0).
    //    - The 2nd MUST be dropped because slots 1, 2, and 3 are still in-flight on the NPU!
    spi[0] = 55;
    pipe.on_spi_dma_complete_isr(spi, 500);

    spi[0] = 66;
    pipe.on_spi_dma_complete_isr(spi, 600);

    assert(!g_mock.in_flight_overwrite.load(std::memory_order_acquire) &&
           "[FAIL] Test 1: SPI wrote into a slot that was still in-flight on the NPU!");
    assert(g_mock.projection_calls == 5 &&
           "[FAIL] Test 1: A single on_npu_dma_complete_isr() freed more than one slot!");

    // Clean up remaining slots
    for (int i = 1; i < 4; ++i) {
        pipe.on_npu_dma_complete_isr();
        g_mock.slot_in_flight[i]--;
    }
    pipe.process_and_dispatch_to_npu();
    pipe.on_npu_dma_complete_isr();
    g_mock.slot_in_flight[0]--;

    std::cout << "[PASS] Test 1\n";
}

// Test 2: High-contention concurrent producer + consumer + NPU completion
void test_concurrent_pipeline_stress() {
    SensorNpuPipeline<4> pipe;
    g_mock.reset(nullptr);

    constexpr int kIterations = 20000;
    std::atomic<bool> start{false};
    std::atomic<bool> stop{false};

    std::thread worker([&]() {
        while (!start.load(std::memory_order_acquire)) {}
        while (!stop.load(std::memory_order_acquire)) {
            pipe.process_and_dispatch_to_npu();
            // Simulate NPU hardware completing 1 DMA descriptor per dispatch
            pipe.on_npu_dma_complete_isr();
        }
    });

    std::thread spi_isr([&]() {
        while (!start.load(std::memory_order_acquire)) {}
        uint8_t data[4] = {1, 2, 3, 4};
        for (int i = 1; i <= kIterations; ++i) {
            data[0] = static_cast<uint8_t>(i & 0xFF);
            pipe.on_spi_dma_complete_isr(data, static_cast<uint64_t>(i));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        stop.store(true, std::memory_order_release);
        pipe.on_spi_dma_complete_isr(data, 999999);
    });

    start.store(true, std::memory_order_release);
    spi_isr.join();
    worker.join();

    assert(g_mock.dma_submits > 0);
    std::cout << "[PASS] Test 2 (" << g_mock.dma_submits << " frames processed)\n";
}

int main() {
    test_single_npu_completion_per_frame();
    test_concurrent_pipeline_stress();
    std::cout << "\nALL V2 TESTS PASSED!\n";
    return 0;
}

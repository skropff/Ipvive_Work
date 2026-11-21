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
    std::vector<float> submitted_coords;
    std::atomic<bool> npu_busy{false};
    std::atomic<bool> data_corruption_detected{false};
    SensorNpuPipeline<4>* pipeline{nullptr};

    void reset(SensorNpuPipeline<4>* p) {
        projection_calls = 0;
        dma_submits = 0;
        submitted_ptrs.clear();
        submitted_timestamps.clear();
        submitted_coords.clear();
        npu_busy = false;
        data_corruption_detected = false;
        pipeline = p;
    }
} g_mock;

extern "C" {
    void project_spi_to_hyperbolic(const uint8_t* raw_spi_dma, ManifoldFrame* out_frame) {
        g_mock.projection_calls.fetch_add(1, std::memory_order_relaxed);
        out_frame->coordinates[0] = static_cast<float>(raw_spi_dma[0]);
    }

    void npu_submit_dma_descriptor(const ManifoldFrame* frame_ptr) {
        g_mock.npu_busy.store(true, std::memory_order_release);
        g_mock.submitted_ptrs.push_back(frame_ptr);
        g_mock.submitted_timestamps.push_back(frame_ptr->timestamp_ns);
        g_mock.submitted_coords.push_back(frame_ptr->coordinates[0]);
        g_mock.dma_submits.fetch_add(1, std::memory_order_release);
    }
}

// Test A: Verifies that slots remain locked while NPU DMA is still in-flight
void test_inflight_dma_protection() {
    SensorNpuPipeline<4> pipe;
    g_mock.reset(&pipe);
    uint8_t spi[4] = {10, 0, 0, 0};

    // 1. Fill all 4 slots
    for (uint64_t i = 1; i <= 4; ++i) {
        spi[0] = static_cast<uint8_t>(i * 10);
        pipe.on_spi_dma_complete_isr(spi, i * 100);
    }

    // 2. Dispatch all 4 to NPU, WITHOUT calling on_npu_dma_complete_isr() yet!
    // The NPU is currently reading from all 4 buffer slots.
    pipe.process_and_dispatch_to_npu();
    assert(g_mock.dma_submits == 4);

    // 3. New SPI interrupt arrives while NPU has NOT finished reading any slot.
    // Since all 4 slots are actively owned by NPU DMA, this MUST be dropped.
    spi[0] = 99;
    pipe.on_spi_dma_complete_isr(spi, 999);

    assert(g_mock.projection_calls == 4 &&
           "[FAIL] Test A: SPI overwrote a buffer slot while NPU DMA was still reading it!");

    // Clean up NPU completions
    for (int i = 0; i < 4; ++i) {
        pipe.on_npu_dma_complete_isr();
    }
    std::cout << "[PASS] Test A\n";
}

// Test B: Verifies strict FIFO ordering across multiple dispatch cycles
void test_strict_fifo_ordering() {
    SensorNpuPipeline<4> pipe;
    g_mock.reset(&pipe);
    uint8_t spi[4] = {0, 0, 0, 0};

    // Cycle 1: Push 2 frames, dispatch both, complete both
    spi[0] = 1; pipe.on_spi_dma_complete_isr(spi, 100);
    spi[0] = 2; pipe.on_spi_dma_complete_isr(spi, 200);
    pipe.process_and_dispatch_to_npu();
    pipe.on_npu_dma_complete_isr();
    pipe.on_npu_dma_complete_isr();

    // Cycle 2: Push 2 more frames, dispatch both, complete both
    spi[0] = 3; pipe.on_spi_dma_complete_isr(spi, 300);
    spi[0] = 4; pipe.on_spi_dma_complete_isr(spi, 400);
    pipe.process_and_dispatch_to_npu();
    pipe.on_npu_dma_complete_isr();
    pipe.on_npu_dma_complete_isr();

    // Cycle 3: Push 1 frame while slot 0 is occupied, then push 2 more
    spi[0] = 5; pipe.on_spi_dma_complete_isr(spi, 500);
    spi[0] = 6; pipe.on_spi_dma_complete_isr(spi, 600);
    pipe.process_and_dispatch_to_npu();
    pipe.on_npu_dma_complete_isr();
    pipe.on_npu_dma_complete_isr();

    assert(g_mock.submitted_timestamps.size() == 6);
    for (size_t i = 0; i < g_mock.submitted_timestamps.size(); ++i) {
        uint64_t expected_ts = (i + 1) * 100;
        assert(g_mock.submitted_timestamps[i] == expected_ts &&
               "[FAIL] Test B: Out-of-order frame dispatch detected!");
    }
    std::cout << "[PASS] Test B\n";
}

// Test C: Verifies worker thread blocks/waits deterministically when idle
// instead of burning 100% CPU in a spin-loop when no frames are ready
void test_worker_blocks_when_empty() {
    SensorNpuPipeline<4> pipe;
    g_mock.reset(&pipe);

    std::atomic<int> loop_iterations{0};
    std::atomic<bool> stop{false};

    std::thread worker([&]() {
        while (!stop.load(std::memory_order_acquire)) {
            pipe.process_and_dispatch_to_npu();
            loop_iterations.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // Let worker run with NO SPI frames arriving for 50ms
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    int idle_iters = loop_iterations.load(std::memory_order_relaxed);

    // Send 1 frame to unblock worker if it is properly waiting
    uint8_t spi[4] = {7, 0, 0, 0};
    stop.store(true, std::memory_order_release);
    pipe.on_spi_dma_complete_isr(spi, 777);
    worker.join();
    pipe.on_npu_dma_complete_isr();

    assert(idle_iters < 10 &&
           "[FAIL] Test C: process_and_dispatch_to_npu() returned immediately on empty queue (busy-spin) instead of waiting!");
    std::cout << "[PASS] Test C\n";
}

// Test D: Verifies ring buffer round-robin slot reuse under continuous single-frame flow
void test_ring_buffer_slot_rotation() {
    SensorNpuPipeline<4> pipe;
    g_mock.reset(&pipe);
    uint8_t spi[4] = {1, 0, 0, 0};

    for (uint64_t i = 0; i < 8; ++i) {
        pipe.on_spi_dma_complete_isr(spi, i);
        pipe.process_and_dispatch_to_npu();
        pipe.on_npu_dma_complete_isr();
    }

    assert(g_mock.submitted_ptrs.size() == 8);
    // Consecutive frames should advance through slots 0, 1, 2, 3 rather than hammering slot 0
    assert(g_mock.submitted_ptrs[0] != g_mock.submitted_ptrs[1] &&
           "[FAIL] Test D: Pipeline is not rotating across ring buffer slots!");
    assert(g_mock.submitted_ptrs[0] == g_mock.submitted_ptrs[4] &&
           "[FAIL] Test D: Pipeline did not wrap around cleanly after Capacity frames!");
    std::cout << "[PASS] Test D\n";
}

int main() {
    test_inflight_dma_protection();
    test_strict_fifo_ordering();
    test_worker_blocks_when_empty();
    test_ring_buffer_slot_rotation();
    std::cout << "\nALL BUG-TRIGGER TESTS PASSED!\n";
    return 0;
}

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
    std::vector<uint64_t> submitted_timestamps;
    std::atomic<bool> out_of_order{false};
    uint64_t last_ts{0};

    void reset() {
        projection_calls = 0;
        dma_submits = 0;
        submitted_timestamps.clear();
        out_of_order = false;
        last_ts = 0;
    }
} g_mock;

extern "C" {
    void project_spi_to_hyperbolic(const uint8_t* raw_spi_dma, ManifoldFrame* out_frame) {
        g_mock.projection_calls.fetch_add(1, std::memory_order_relaxed);
        out_frame->coordinates[0] = static_cast<float>(raw_spi_dma[0]);
    }

    void npu_submit_dma_descriptor(const ManifoldFrame* frame_ptr) {
        // Simulate realistic NPU descriptor submission delay to widen thread interleaving windows
        std::this_thread::yield();
        uint64_t ts = frame_ptr->timestamp_ns;
        if (ts < g_mock.last_ts) {
            g_mock.out_of_order.store(true, std::memory_order_release);
        }
        g_mock.last_ts = ts;
        g_mock.submitted_timestamps.push_back(ts);
        g_mock.dma_submits.fetch_add(1, std::memory_order_release);
    }
}

// Test 1: Multiple worker threads racing on process_and_dispatch_to_npu()
// with delayed NPU DMA completions
void test_multi_worker_race_and_delayed_npu_irq() {
    SensorNpuPipeline<4> pipe;
    g_mock.reset();

    constexpr int kNumWorkers = 4;
    constexpr int kFramesPerBatch = 4;
    constexpr int kBatches = 200;

    std::atomic<bool> stop{false};
    std::atomic<int> batch_dispatched{0};

    std::vector<std::thread> workers;
    for (int w = 0; w < kNumWorkers; ++w) {
        workers.emplace_back([&]() {
            while (!stop.load(std::memory_order_acquire)) {
                pipe.process_and_dispatch_to_npu();
                batch_dispatched.fetch_add(1, std::memory_order_release);
            }
        });
    }

    uint8_t spi[4] = {1, 0, 0, 0};
    uint64_t ts = 1;

    for (int b = 0; b < kBatches; ++b) {
        int before = batch_dispatched.load(std::memory_order_acquire);

        // Produce 4 frames
        for (int i = 0; i < kFramesPerBatch; ++i) {
            spi[0] = static_cast<uint8_t>(ts & 0xFF);
            pipe.on_spi_dma_complete_isr(spi, ts++);
        }

        // Wait until all 4 workers have dispatched their frames to NPU
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (batch_dispatched.load(std::memory_order_acquire) - before < kFramesPerBatch) {
            if (std::chrono::steady_clock::now() > deadline) {
                std::cerr << "\n[FAIL] Test 1: Deadlock or lost wake-up detected at batch " << b << "!\n";
                std::abort();
            }
            std::this_thread::yield();
        }

        // Complete all 4 NPU DMA reads AFTER all 4 have been submitted
        for (int i = 0; i < kFramesPerBatch; ++i) {
            pipe.on_npu_dma_complete_isr();
        }
    }

    stop.store(true, std::memory_order_release);
    // Wake all 4 workers so they can exit cleanly
    for (int w = 0; w < kNumWorkers; ++w) {
        pipe.on_spi_dma_complete_isr(spi, ts++);
    }
    for (auto& t : workers) {
        t.join();
    }
    for (int w = 0; w < kNumWorkers; ++w) {
        pipe.on_npu_dma_complete_isr();
    }

    assert(!g_mock.out_of_order.load(std::memory_order_acquire) &&
           "[FAIL] Test 1: Out-of-order dispatch or data corruption under multi-worker contention!");
    std::cout << "[PASS] Test 1 (" << g_mock.dma_submits << " frames processed cleanly)\n";
}

int main() {
    test_multi_worker_race_and_delayed_npu_irq();
    std::cout << "\nALL V3 TESTS PASSED!\n";
    return 0;
}

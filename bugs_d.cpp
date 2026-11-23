#include "pipeline.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

struct MockHardware {
    std::atomic<int> dma_submits{0};
    std::atomic<int> projection_calls{0};
    std::vector<uint64_t> dispatched_timestamps;
    std::atomic<bool> hold_in_submit{false};
    std::atomic<int> threads_in_submit{0};

    void reset() {
        dma_submits = 0;
        projection_calls = 0;
        dispatched_timestamps.clear();
        hold_in_submit = false;
        threads_in_submit = 0;
    }
} g_mock;

extern "C" {
    void project_spi_to_hyperbolic(const uint8_t* raw_spi_dma, ManifoldFrame* out_frame) {
        g_mock.projection_calls.fetch_add(1, std::memory_order_relaxed);
        out_frame->coordinates[0] = static_cast<float>(raw_spi_dma[0]);
    }

    void npu_submit_dma_descriptor(const ManifoldFrame* frame_ptr) {
        g_mock.threads_in_submit.fetch_add(1, std::memory_order_release);
        while (g_mock.hold_in_submit.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        g_mock.dispatched_timestamps.push_back(frame_ptr->timestamp_ns);
        g_mock.dma_submits.fetch_add(1, std::memory_order_release);
    }
}

// Test 1: Multiple workers claiming indices BEFORE frames arrive,
// while one worker stalls inside npu_submit_dma_descriptor
void test_worker_preclaim_and_stall() {
    SensorNpuPipeline<4> pipe;
    g_mock.reset();

    // Hold Worker 0 inside npu_submit_dma_descriptor so it cannot advance the next worker yet
    g_mock.hold_in_submit.store(true, std::memory_order_release);

    std::atomic<int> finished_workers{0};
    std::vector<std::thread> workers;

    // Spawn 5 workers (more than Capacity = 4) when queue is completely empty
    for (int w = 0; w < 5; ++w) {
        workers.emplace_back([&]() {
            pipe.process_and_dispatch_to_npu();
            finished_workers.fetch_add(1, std::memory_order_release);
        });
        // Give each worker time to enter and block in sequence
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    uint8_t spi[4] = {1, 0, 0, 0};

    // Push 4 frames (fills all 4 slots)
    for (uint64_t i = 1; i <= 4; ++i) {
        spi[0] = static_cast<uint8_t>(i);
        pipe.on_spi_dma_complete_isr(spi, i * 100);
    }

    // Wait until Worker 0 reaches npu_submit_dma_descriptor
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    while (g_mock.threads_in_submit.load(std::memory_order_acquire) == 0) {
        if (std::chrono::steady_clock::now() > deadline) {
            std::cerr << "\n[FAIL] Worker 0 never reached npu_submit_dma_descriptor!\n";
            std::abort();
        }
        std::this_thread::yield();
    }

    // Release Worker 0 so frames 1..4 can dispatch
    g_mock.hold_in_submit.store(false, std::memory_order_release);

    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (finished_workers.load(std::memory_order_acquire) < 4) {
        if (std::chrono::steady_clock::now() > deadline) {
            std::cerr << "\n[FAIL] Deadlock waiting for first 4 workers!\n";
            std::abort();
        }
        std::this_thread::yield();
    }

    // Complete all 4 NPU DMA reads
    for (int i = 0; i < 4; ++i) {
        pipe.on_npu_dma_complete_isr();
    }

    // Now push a 5th frame for the 5th waiting worker
    spi[0] = 5;
    pipe.on_spi_dma_complete_isr(spi, 500);

    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (finished_workers.load(std::memory_order_acquire) < 5) {
        if (std::chrono::steady_clock::now() > deadline) {
            std::cerr << "\n[FAIL] 5th worker deadlocked!\n";
            std::abort();
        }
        std::this_thread::yield();
    }
    pipe.on_npu_dma_complete_isr();

    for (auto& t : workers) t.join();

    assert(g_mock.dispatched_timestamps.size() == 5);
    for (size_t i = 0; i < 5; ++i) {
        assert(g_mock.dispatched_timestamps[i] == (i + 1) * 100 &&
               "[FAIL] Out-of-order timestamp dispatch!");
    }
    std::cout << "[PASS] Test 1\n";
}

// Test 2: High-iteration continuous multi-worker stress test (100,000 frames)
void test_heavy_multi_worker_fuzz() {
    SensorNpuPipeline<4> pipe;
    g_mock.reset();

    constexpr int kWorkers = 3;
    constexpr int kTotalFrames = 100000;
    std::atomic<bool> stop{false};
    std::atomic<bool> start{false};

    std::vector<std::thread> workers;
    for (int w = 0; w < kWorkers; ++w) {
        workers.emplace_back([&]() {
            while (!start.load(std::memory_order_acquire)) {}
            while (!stop.load(std::memory_order_acquire)) {
                pipe.process_and_dispatch_to_npu();
                pipe.on_npu_dma_complete_isr();
            }
        });
    }

    std::thread spi_isr([&]() {
        while (!start.load(std::memory_order_acquire)) {}
        uint8_t spi[4] = {1, 2, 3, 4};
        for (int i = 1; i <= kTotalFrames; ++i) {
            spi[0] = static_cast<uint8_t>(i & 0xFF);
            pipe.on_spi_dma_complete_isr(spi, static_cast<uint64_t>(i));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        stop.store(true, std::memory_order_release);
        for (int w = 0; w < kWorkers; ++w) {
            pipe.on_spi_dma_complete_isr(spi, 999999);
        }
    });

    start.store(true, std::memory_order_release);
    spi_isr.join();
    for (auto& t : workers) t.join();

    assert(g_mock.dma_submits > 0);
    std::cout << "[PASS] Test 2 (" << g_mock.dma_submits << " frames processed cleanly)\n";
}

int main() {
    test_worker_preclaim_and_stall();
    test_heavy_multi_worker_fuzz();
    std::cout << "\nALL V5 TESTS PASSED!\n";
    return 0;
}

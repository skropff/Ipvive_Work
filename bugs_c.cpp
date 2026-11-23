#include "pipeline.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

struct MockHardware {
    std::atomic<int> dma_submits{0};
    std::atomic<int> active_npu_reads{0};
    std::atomic<bool> premature_slot_reuse{false};
    std::atomic<int> slot_in_use[4]{};

    void reset() {
        dma_submits = 0;
        active_npu_reads = 0;
        premature_slot_reuse = false;
        for (int i = 0; i < 4; ++i) slot_in_use[i] = 0;
    }
} g_mock;

extern "C" {
    void project_spi_to_hyperbolic(const uint8_t* raw_spi_dma, ManifoldFrame* out_frame) {
        out_frame->coordinates[0] = static_cast<float>(raw_spi_dma[0]);
    }

    void npu_submit_dma_descriptor(const ManifoldFrame* frame_ptr) {
        g_mock.dma_submits.fetch_add(1, std::memory_order_release);
    }
}

// Test 1: Multiple workers waiting when fewer frames arrive than waiting threads,
// followed by staggered arrivals and completions
void test_staggered_multi_worker_interleaving() {
    SensorNpuPipeline<4> pipe;
    g_mock.reset();

    constexpr int kWorkers = 4;
    std::atomic<int> completed_calls{0};
    std::atomic<bool> stop{false};

    std::vector<std::thread> workers;
    for (int w = 0; w < kWorkers; ++w) {
        workers.emplace_back([&]() {
            while (!stop.load(std::memory_order_acquire)) {
                pipe.process_and_dispatch_to_npu();
                completed_calls.fetch_add(1, std::memory_order_release);
            }
        });
    }

    // Give all 4 workers time to enter process_and_dispatch_to_npu() and block
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    uint8_t spi[4] = {1, 0, 0, 0};

    // Feed 1 frame at a time across 20 cycles while 4 workers are contending
    for (uint64_t i = 1; i <= 20; ++i) {
        int before = completed_calls.load(std::memory_order_acquire);
        spi[0] = static_cast<uint8_t>(i);
        pipe.on_spi_dma_complete_isr(spi, i * 10);

        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
        while (completed_calls.load(std::memory_order_acquire) == before) {
            if (std::chrono::steady_clock::now() > deadline) {
                std::cerr << "\n[FAIL] Test 1: Lost wake-up or deadlock on frame " << i << "!\n";
                std::abort();
            }
            std::this_thread::yield();
        }
        pipe.on_npu_dma_complete_isr();
    }

    stop.store(true, std::memory_order_release);
    for (int w = 0; w < kWorkers; ++w) {
        pipe.on_spi_dma_complete_isr(spi, 999);
    }
    for (auto& t : workers) t.join();
    for (int w = 0; w < kWorkers; ++w) {
        pipe.on_npu_dma_complete_isr();
    }

    std::cout << "[PASS] Test 1 (" << g_mock.dma_submits << " staggered frames dispatched cleanly)\n";
}

// Test 2: High-speed NPU completion racing with SPI ISR and worker thread
void test_rapid_completion_race() {
    SensorNpuPipeline<4> pipe;
    g_mock.reset();

    constexpr int kFrames = 50000;
    std::atomic<bool> start{false};
    std::atomic<bool> stop{false};

    std::thread worker([&]() {
        while (!start.load(std::memory_order_acquire)) {}
        while (!stop.load(std::memory_order_acquire)) {
            pipe.process_and_dispatch_to_npu();
            pipe.on_npu_dma_complete_isr();
        }
    });

    std::thread spi_isr([&]() {
        while (!start.load(std::memory_order_acquire)) {}
        uint8_t data[4] = {1, 2, 3, 4};
        for (int i = 1; i <= kFrames; ++i) {
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
    std::cout << "[PASS] Test 2 (" << g_mock.dma_submits << " rapid frames completed without deadlock)\n";
}

int main() {
    test_staggered_multi_worker_interleaving();
    test_rapid_completion_race();
    std::cout << "\nALL V4 TESTS PASSED!\n";
    return 0;
}

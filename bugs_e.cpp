#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <chrono>
#include <cassert>
#include <cstring>
#include <atomic>
#include <future>

// Include your implementation header here
#include "pipeline.hpp"

// --- Test Harness & Hardware Mock State ---
struct SubmittedJob {
    const ManifoldFrame* ptr;
    uint64_t snapshot_timestamp;
    float snapshot_first_coord;
};

std::mutex hw_queue_mtx;
std::queue<SubmittedJob> npu_hw_queue;
std::atomic<bool> block_npu_completions{false};

extern "C" {
    void project_spi_to_hyperbolic(const uint8_t* raw_spi_dma, ManifoldFrame* out_frame) {
        // Simulate SPI DMA projection work
        uint64_t val = 0;
        std::memcpy(&val, raw_spi_dma, sizeof(uint64_t));
        out_frame->coordinates[0] = static_cast<float>(val);
    }

    void npu_submit_dma_descriptor(const ManifoldFrame* frame_ptr) {
        // Snapshot the memory at the exact moment the NPU receives the pointer
        std::lock_guard<std::mutex> lock(hw_queue_mtx);
        npu_hw_queue.push({
            frame_ptr,
            frame_ptr->timestamp_ns,
            frame_ptr->coordinates[0]
        });
    }
}

void reset_mock_hardware() {
    std::lock_guard<std::mutex> lock(hw_queue_mtx);
    while (!npu_hw_queue.empty()) npu_hw_queue.pop();
    block_npu_completions = false;
}

// -------------------------------------------------------------------------
// TEST 1: Burst Saturation & Out-of-Order NPU Execution
// Simulates SPI bursts filling the ring buffer while NPU completions lag,
// followed by continuous high-frequency streaming.
// -------------------------------------------------------------------------
void test_burst_and_stream_ordering() {
    std::cout << "[Running Test 1: Burst Saturation & Ordering]...\n";
    reset_mock_hardware();
    SensorNpuPipeline<4> pipeline;
    std::atomic<bool> stop_flag{false};

    // Real-time worker thread
    std::thread worker([&]() {
        while (!stop_flag.load()) {
            pipeline.process_and_dispatch_to_npu();
        }
    });

    // Simulated NPU DMA completion ISR thread
    std::thread npu_isr([&]() {
        while (!stop_flag.load()) {
            bool has_job = false;
            {
                std::lock_guard<std::mutex> lock(hw_queue_mtx);
                if (!npu_hw_queue.empty() && !block_npu_completions.load()) {
                    npu_hw_queue.pop();
                    has_job = true;
                }
            }
            if (has_job) {
                pipeline.on_npu_dma_complete_isr();
            } else {
                std::this_thread::yield();
            }
        }
    });

    // Fire 10,000 SPI interrupts with occasional NPU stalls
    uint8_t dummy_spi[8] = {0};
    for (uint64_t seq = 1; seq <= 10000; ++seq) {
        std::memcpy(dummy_spi, &seq, sizeof(seq));
        
        // Periodically stall NPU completions to force buffer full conditions
        if (seq % 17 == 0) block_npu_completions = true;
        if (seq % 23 == 0) block_npu_completions = false;

        pipeline.on_spi_dma_complete_isr(dummy_spi, seq * 100);
        
        if (seq % 5 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(5));
        }
    }

    block_npu_completions = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stop_flag = true;

    // Unblock worker if it is waiting on a semaphore at shutdown
    for (int i = 0; i < 4; ++i) {
        pipeline.on_spi_dma_complete_isr(dummy_spi, 999999);
    }
    worker.join();
    npu_isr.join();
    std::cout << "  -> Test 1 completed without crash.\n";
}

// -------------------------------------------------------------------------
// TEST 2: Multi-Worker Contention & Shutdown Deadlock Check
// Runs multiple worker threads pulling from the pipeline under load.
// -------------------------------------------------------------------------
/*
void test_multi_worker_contention() {
    std::cout << "[Running Test 2: Multi-Worker Contention]...\n";
    reset_mock_hardware();
    SensorNpuPipeline<4> pipeline;
    std::atomic<int> processed_count{0};
    const int target_frames = 500;

    auto worker_fn = [&]() {
        while (processed_count.load() < target_frames) {
            pipeline.process_and_dispatch_to_npu();
            processed_count.fetch_add(1);
        }
    };

    // Launch 2 worker threads
    std::thread w1(worker_fn);
    std::thread w2(worker_fn);

    // Feed SPI frames and immediately complete NPU DMA
    std::thread spi_and_npu([&]() {
        uint8_t dummy_spi[8] = {0};
        uint64_t seq = 1;
        while (processed_count.load() < target_frames) {
            std::memcpy(dummy_spi, &seq, sizeof(seq));
            pipeline.on_spi_dma_complete_isr(dummy_spi, seq);
            seq++;

            bool completed = false;
            {
                std::lock_guard<std::mutex> lock(hw_queue_mtx);
                if (!npu_hw_queue.empty()) {
                    npu_hw_queue.pop();
                    completed = true;
                }
            }
            if (completed) {
                pipeline.on_npu_dma_complete_isr();
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    });

    // Timeout watchdog to detect deadlocks
    auto future = std::async(std::launch::async, [&]() {
        spi_and_npu.join();
        w1.join();
        w2.join();
    });

    if (future.wait_for(std::chrono::seconds(3)) == std::future_status::timeout) {
        std::cerr << "  [FAIL] Test 2 DEADLOCKED! Workers hung indefinitely.\n";
        std::exit(1);
    }
    std::cout << "  -> Test 2 completed.\n";
}
*/
// -------------------------------------------------------------------------
// TEST 3: Data Integrity / Premature Slot Overwrite Check
// Verifies that the NPU hardware never sees a buffer overwritten while 
// the NPU is still actively reading from that DMA address.
// -------------------------------------------------------------------------
void test_dma_data_integrity() {
    std::cout << "[Running Test 3: DMA Slot Overwrite & Data Integrity]...\n";
    reset_mock_hardware();
    SensorNpuPipeline<4> pipeline;
    std::atomic<bool> done{false};
    std::atomic<int> corruption_errors{0};

    std::thread worker([&]() {
        while (!done.load()) {
            pipeline.process_and_dispatch_to_npu();
        }
    });

    // Slow NPU consumer that checks memory *after* holding the pointer for a bit
    std::thread slow_npu([&]() {
        int consumed = 0;
        while (consumed < 200) {
            SubmittedJob job{};
            bool has_job = false;
            {
                std::lock_guard<std::mutex> lock(hw_queue_mtx);
                if (!npu_hw_queue.empty()) {
                    job = npu_hw_queue.front();
                    npu_hw_queue.pop();
                    has_job = true;
                }
            }

            if (has_job) {
                // Simulate NPU taking time to read DMA memory over PCIe/AXI bus
                std::this_thread::sleep_for(std::chrono::microseconds(150));

                // Re-read the live buffer_pool_ memory BEFORE signaling completion
                uint64_t live_ts = job.ptr->timestamp_ns;
                float live_coord = job.ptr->coordinates[0];

                if (live_ts != job.snapshot_timestamp || live_coord != job.snapshot_first_coord) {
                    corruption_errors.fetch_add(1);
                }

                pipeline.on_npu_dma_complete_isr();
                consumed++;
            }
        }
        done = true;
    });

    // Rapid SPI producer
    uint8_t dummy_spi[8] = {0};
    uint64_t seq = 1;
    while (!done.load()) {
        std::memcpy(dummy_spi, &seq, sizeof(seq));
        pipeline.on_spi_dma_complete_isr(dummy_spi, seq);
        seq++;
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }

    slow_npu.join();
    // Unblock worker to exit cleanly
    for (int i = 0; i < 4; ++i) pipeline.on_spi_dma_complete_isr(dummy_spi, 0);
    worker.join();

    if (corruption_errors.load() > 0) {
        std::cerr << "  [FAIL] Test 3 DETECTED DATA CORRUPTION! Buffer overwritten "
                  << corruption_errors.load() << " times while NPU was reading it.\n";
    } else {
        std::cout << "  -> Test 3 passed.\n";
    }
}

// -------------------------------------------------------------------------
// TEST 4: Hardware Alignment & Size Verification
// Checks cache-line alignment and struct padding assumptions.
// -------------------------------------------------------------------------
void test_memory_layout() {
    std::cout << "[Running Test 4: Memory Layout & Alignment]...\n";
    std::cout << "  sizeof(ManifoldFrame): " << sizeof(ManifoldFrame) << " bytes\n";
    std::cout << "  alignof(ManifoldFrame): " << alignof(ManifoldFrame) << " bytes\n";
    
    if (sizeof(ManifoldFrame) != 64) {
        std::cout << "  [WARNING] ManifoldFrame spans multiple cache lines (" 
                  << sizeof(ManifoldFrame) << " bytes).\n";
    }
}

int main() {
    test_memory_layout();
    test_dma_data_integrity();
    test_burst_and_stream_ordering();
    test_multi_worker_contention();
    return 0;
}

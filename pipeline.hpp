#include <array>
#include <atomic>
#include <cstdint>
#include <semaphore>
#include <cstddef>
#include <optional>

// 64-byte aligned payload for NPU DMA compatibility
struct alignas(64) ManifoldFrame {
    uint64_t timestamp_ns;
    float coordinates[16]; // 3D hyperbolic manifold tensor representation
};

// Mock Hardware Interfaces (Do not implement)
extern "C" {
    // Reads raw SPI DMA bytes and projects them into the target frame in-place
    void project_spi_to_hyperbolic(const uint8_t* raw_spi_dma, ManifoldFrame* out_frame);
    
    // Submits a physical/pinned memory pointer directly to the EdgeCortix NPU queue
    void npu_submit_dma_descriptor(const ManifoldFrame* frame_ptr);
}

template <size_t Capacity = 4>
class SensorNpuPipeline {
private:
    // Pre-allocated, cache-aligned memory pool
    alignas(64) std::array<ManifoldFrame, Capacity> buffer_pool_{};

    // TODO: Define your synchronization primitives (semaphores, atomic indices)
    // Hint: Track both free slots available for SPI and ready slots for the NPU.
	
	// Declares Capacity empty slots without constructing any semaphores
	std::array<std::optional<std::binary_semaphore>, Capacity> spi_sem;
	
	// Declares Capacity empty slots without constructing any semaphores
	std::array<std::optional<std::binary_semaphore>, Capacity> npu_sem;
	
	std::array<std::optional<std::counting_semaphore<Capacity>> extra_npu_sem;
	
	//std::array<std::optional<std::binary_semaphore>, Capacity> final_sem;

public:
    SensorNpuPipeline() {
        // TODO: Initialize semaphores and state
		for (int i = 0; i < Capacity; i = i + 1) {
			spi_sem.emplace(1);
		}
		
		for (int i = 0; i < Capacity; i = i + 1) {
			npu_sem.emplace(0);
		}
		
		extra_npu_sem.emplace(0);
		
		/*
		for (int i = 0; i < Capacity; i = i + 1) {
			final_sem.emplace(0);
		}
		*/
    }

    // 1. CALLED FROM SPI INTERRUPT CONTEXT (ISR)
    // Must be non-blocking (O(1) deterministic). If the NPU is stalled and no
    // slots are free, drop the oldest or newest frame without blocking the ISR.
    void on_spi_dma_complete_isr(const uint8_t* raw_spi_data, uint64_t timestamp) noexcept {
        // TODO: Acquire a free slot without blocking, write directly into
        // buffer_pool_, and signal the worker/NPU thread.
		bool success = false;
		int match;
		
		for (int i = 0; i < Capacity; i = i + 1) {
			//Try to acquire next SPI sempahore. 
			if (spi_sem[i].try_acquire()) {
				match = i;
				success = true;
				break;
			}
		}
		
		if (success) {
			project_spi_to_hyperbolic(raw_spi_data, buffer_pool_ + match);
			npu_sem[match].release();
			extra_npu_sem[0].release();
		}
    }

    // 2. CALLED FROM PINNED REAL-TIME WORKER THREAD (SCHED_FIFO)
    // Waits deterministically for a ready frame, hands the pointer to the NPU,
    // and manages buffer lifecycle.
    void process_and_dispatch_to_npu() noexcept {
        // TODO: Wait on ready semaphore, dispatch pointer to NPU, and handle slot release.
		int starting = 0;
		int index;
		while (true) {
			extra_npu_sem[0].acquire();
			
			for (int i = 0; i < Capacity; i = i + 1) {
				index = (i + starting) % Capacity;
				if (npu_sem[index].try_acquire()) {
					starting = i + 1;
					npu_submit_dma_descriptor(buffer_pool_ + index);
					break;
				}
			}
			
	
			
			
			
		}
    }

    // 3. CALLED FROM NPU COMPLETION INTERRUPT
    // Releases the buffer slot back to the pool once NPU DMA read finishes.
    void on_npu_dma_complete_isr() noexcept {
        // TODO: Release the slot back to the free semaphore.
    }
};

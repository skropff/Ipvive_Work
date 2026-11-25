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
	
	//std::array<std::optional<std::counting_semaphore<Capacity>> extra_npu_sem;

	//std::array<std::optional<std::counting_semaphore<Capacity>>, 1> extra_npu_sem;

	
	//std::array<std::optional<std::binary_semaphore>, Capacity> final_sem;
	
	//std::array<std::optional<std::binary_semaphore>, 1> interrupt_sem;
	
	//std::array<std::optional<std::counting_semaphore<>>, 1> waiting_count;
	
	//std::array<std::optional<std::binary_semaphore>, 1> is_first;
	
	//std::array<std::optional<std::counting_semaphore<>>, 1> spi_interrupt_count;
	
	std::array<std::optional<std::binary_semaphore>, 1> can_access_atomic;
	
	int atomic_value[1];
	
	int spi_turn[1];
	
	std::array<std::optional<std::binary_semaphore>, Capacity> order_control;
	
	int npu_count[1];
	

public:
    SensorNpuPipeline() {
        // TODO: Initialize semaphores and state
		for (int i = 0; i < Capacity; i = i + 1) {
			spi_sem[i].emplace(1);
		}
		
		for (int i = 0; i < Capacity; i = i + 1) {
			npu_sem[i].emplace(0);
		}
		
		//extra_npu_sem[0].emplace(0);
		
		/*
		for (int i = 0; i < Capacity; i = i + 1) {
			final_sem[i].emplace(0);
		}
		*/
		
		//interrupt_sem[0].emplace(0);
		
		//waiting_count[0].emplace(0);
		
		//is_first[0].emplace(0);
		
		//spi_interrupt_count[0].emplace(0);
		
		can_access_atomic[0].emplace(1);
		
		atomic_value[0] = 0;
		
		spi_turn[0] = 0;
		/*
		for (int i = 0; i < Capacity; i = i + 1) {
			order_control[i].emplace(1);
		}
		*/
		
		order_control[0].emplace(1);
		
		for (int i = 1; i < Capacity; i = i + 1) {
			order_control[i].emplace(0);
		}
		
		npu_count[0] = 0;
    }

    // 1. CALLED FROM SPI INTERRUPT CONTEXT (ISR)
    // Must be non-blocking (O(1) deterministic). If the NPU is stalled and no
    // slots are free, drop the oldest or newest frame without blocking the ISR.
    void on_spi_dma_complete_isr(const uint8_t* raw_spi_data, uint64_t timestamp) noexcept {
        // TODO: Acquire a free slot without blocking, write directly into
        // buffer_pool_, and signal the worker/NPU thread.
		bool success = false;
		int match;
		//printf("x1\n");
		/*
		for (int i = 0; i < Capacity; i = i + 1) {
			//Try to acquire next SPI sempahore. 
			if (spi_sem[i]->try_acquire()) {
				match = i;
				success = true;
				break;
			}
		}
		*/
		
		match = spi_turn[0];
		if (spi_sem[match]->try_acquire()) {
			success = true;
			spi_turn[0] = (spi_turn[0] + 1) % Capacity;
		}
		
		if (success) {
			project_spi_to_hyperbolic(raw_spi_data, &buffer_pool_[match]);
			(&buffer_pool_[match])->timestamp_ns = timestamp;
			npu_sem[match]->release();
			//printf("index %d is released\n", match);
			//extra_npu_sem[0]->release();
		}
		
		
		//spi_interrupt_count[0]->release();
    }
	
	/*
	if (!interrupt_sem[0]->try_acquire()) {
		interrupt_sem[0].release();
	}
	else {
		interrupt_sem[0].release();
	}
	*/
	
	//interrupt_sem[0].emplace(1);
	
	//waiting_count[0]->try_acquire();

    // 2. CALLED FROM PINNED REAL-TIME WORKER THREAD (SCHED_FIFO)
    // Waits deterministically for a ready frame, hands the pointer to the NPU,
    // and manages buffer lifecycle.
    void process_and_dispatch_to_npu() noexcept {
        // TODO: Wait on ready semaphore, dispatch pointer to NPU, and handle slot release.
		//waiting_count[0]->
		//interrupt_sem[0]->acquire();
		
		//spi_interrupt_count[0]->acquire();
		//printf("starting\n");
		int starting = 0;
		int index;
		int next;
		/*
		while (extra_npu_sem[0]->try_acquire()) {
			printf("extra npu sem acquired\n");
			for (int i = 0; i < Capacity; i = i + 1) {
				printf("inner loop\n");
				index = (i + starting) % Capacity;
				if (npu_sem[index]->try_acquire()) {
					starting = i + 1;
					npu_submit_dma_descriptor(&buffer_pool_[index]);
					final_sem[index]->release();
					break;
				}
			}
			*/
			
			can_access_atomic[0]->acquire();
			index = atomic_value[0];
			atomic_value[0] = (atomic_value[0] + 1) % Capacity;
			can_access_atomic[0]->release();
			next = (index + 1) % Capacity;
			
			printf("index %d being acquired\n", index);
			npu_sem[index]->acquire();
			printf("bench b\n");
			order_control[index]->acquire();
			npu_submit_dma_descriptor(&buffer_pool_[index]);
			//final_sem[index]->release();
			order_control[next]->release();
			printf("outer loop completed\n");
			
			
			
	}
		
    

    // 3. CALLED FROM NPU COMPLETION INTERRUPT
    // Releases the buffer slot back to the pool once NPU DMA read finishes.
    void on_npu_dma_complete_isr() noexcept {
        // TODO: Release the slot back to the free semaphore.
		//printf("x2\n");
		/*
		for (int i = 0; i < Capacity; i = i + 1) {
			if (final_sem[i]->try_acquire()) {
				spi_sem[i]->release();
			}
		}
		*/
		spi_sem[npu_count[0]]->release();
		npu_count[0] = (npu_count[0] + 1) % Capacity;
		
		
    }
};

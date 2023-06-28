#pragma once
#include "../globals.hpp"
#include "cuda.hpp"

namespace epic
{
    namespace gpu
    {

        struct DeviceStream
        {

            cudaStream_t stream;
            cudaError_t err;
            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEvent_t start_2;
            cudaEvent_t stop_2;
            void create();
            void start_timer();
            void stop_timer();
            float duration_in_millis();
            void synchronize_stream();

            DeviceStream() = default;
            ~DeviceStream();
        };

        DeviceStream::~DeviceStream()
        {
            DEBUG_BEFORE_DESTRUCT("DeviceStream (ALL)");
            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            err = cudaStreamDestroy(stream);
            DEBUG_AFTER_DESTRUCT("DeviceStream (ALL)");
        }

        void DeviceStream::create()
        {
            err = cudaStreamCreate(&stream);
            cudaEventCreateWithFlags(&start, cudaDeviceScheduleBlockingSync);
            cudaEventCreateWithFlags(&stop, cudaDeviceScheduleBlockingSync);
        }

        void DeviceStream::start_timer()
        {
            cudaEventRecord(start, stream);
        }

        void DeviceStream::stop_timer()
        {
            cudaEventRecord(stop, stream);
        }

        float DeviceStream::duration_in_millis()
        {
            float milliseconds;
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            return milliseconds;
        }

        void DeviceStream::synchronize_stream()
        {
            cudaStreamSynchronize(stream);
        }
    }
}
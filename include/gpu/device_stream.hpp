#pragma once
#include "../globals.hpp"
#include "cuda.hpp"
#include <vector>

namespace epic
{
    namespace gpu
    {

        struct DeviceStream
        {

            cudaStream_t stream;
            cudaError_t err;
            std::vector<cudaEvent_t> start_events;
            std::vector<cudaEvent_t> stop_events;

            // cudaEvent_t start;
            // cudaEvent_t stop;
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
            for (int i = 0; i < start_events.size(); i++)
            {
                cudaEventDestroy(start_events.pop_back());
            }
            for (int i = 0; i < stop_events.size(); i++)
            {
                cudaEventDestroy(stop_events.pop_back());
            }
            //          cudaEventDestroy(start);
            //          cudaEventDestroy(stop);

            err = cudaStreamDestroy(stream);
            DEBUG_AFTER_DESTRUCT("DeviceStream (ALL)");
        }

        void DeviceStream::create()
        {
            err = cudaStreamCreate(&stream);
            //          cudaEventCreate(&start);
            //          cudaEventCreate(&stop);
        }

        void DeviceStream::start_timer()
        {
            cudaEvent_t start;
            cudaEventCreate(&start);
            cudaEvent_t stop;
            cudaEventCreate(&stop);
            start_events.push_back(start);
            stop_events.push_back(stop);
            cudaEventRecord(start, stream);
        }

        void DeviceStream::stop_timer()
        {
            cudaEventRecord(stop_events.back(), stream);
        }

        float DeviceStream::duration_in_millis()
        {
            float milliseconds;
            cudaEventSynchronize(stop_events.back());
            cudaEventElapsedTime(&milliseconds, start_events.pop_back(), stop_events.pop_back());
            return milliseconds;
        }

        void DeviceStream::synchronize_stream()
        {
            cudaStreamSynchronize(stream);
        }
    }
}
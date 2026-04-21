# TD-MPC2 Benchmarking Log

## DMControl Walker-Walk (State-based)
- **Status**: Verified implementation health.
- **Results**:
    - **Steps**: 41,500
    - **Reward (Max)**: 990.7
    - **Consistency**: High. The agent reached ~950+ reward within 20k steps and stabilized.
- **Config**:
    - `batch_size`: 256
    - `num_samples`: 512
    - `iterations`: 6
    - `horizon`: 5
- **Conclusion**: The TD-MPC2 implementation is correctly handling state-based DMControl environments with the current patches. Performance matches paper expectations for "fast solve" on simple tasks.

## SIT Integration Experiments
- **CarRacing-v3**:
    - **Phase 1**: Reached ~445 reward at 100k steps.
    - **Resumption Issues**: Buffer sampling and CUDA storage issues resolved.
    - **Current Blockers**: Underperformance on 1M step scale due to gradient explosion with high batch sizes on fresh buffers.

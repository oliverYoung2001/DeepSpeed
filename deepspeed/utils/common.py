import torch

def cuda_memory_analyze(step=0, print_mm_suage=False):
    """
    Useful utility functions migrated from deepseped.
    """

    # global n_caching_allocator_flushes

    g_rank = torch.distributed.get_rank()
    # tp_rank = mpu.get_tensor_model_parallel_rank()
    # pp_rank = mpu.get_pipeline_model_parallel_rank()
    # dp_rank = mpu.get_data_parallel_rank()

    rank_id = f"Rank:{g_rank}"#-tp{tp_rank}-pp{pp_rank}-dp{dp_rank}"

    if print_mm_suage and g_rank == 0:
        print(
            f"{rank_id}: Step {step}: "
            f"Allocated {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),4 )} GB, "
            f"Max_Allocated {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),4)} GB, "
            f"Reserved {round(torch.cuda.memory_reserved()/ (1024 * 1024 * 1024),4)} GB, "
            f"Max_Reserved {round(torch.cuda.max_memory_reserved()/ (1024 * 1024 * 1024),4)} GB "
        )

        torch.cuda.reset_peak_memory_stats()

    # warn user about caching allocator flushes
    memory_stats = torch.cuda.memory_stats()
    alloc_retries = memory_stats.get("num_alloc_retries")
    # if alloc_retries is None:
    #     alloc_retries = 0
    # if alloc_retries > n_caching_allocator_flushes:
    #     retry_count = alloc_retries - n_caching_allocator_flushes
    #     if g_rank == 0:
    #         print(
    #             f"[WARN]: {rank_id}: pytorch allocator cache flushes {retry_count} times since last step."
    #             "this happens when there is high memory pressure and is detrimental to "
    #             "performance. if this is happening frequently consider adjusting "
    #             "settings to reduce memory consumption. If you are unable to "
    #             "make the cache flushes go away consider adding "
    #             "torch.cuda.empty_cache() calls in your training loop to ensure "
    #             "that all ranks flush their caches at the same time"
    #         )
    #     n_caching_allocator_flushes = alloc_retries

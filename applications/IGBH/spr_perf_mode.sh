#!/bin/bash
# Set system to performance mode
/home/files/sudobins/spr_scaling_governor performance
/home/files/sudobins/spr_no_turbo 0
/home/files/sudobins/spr_numa_balancing 0

#Clear cache (should be conducted before each workload run for accurate benchmarking)
/home/files/sudobins/spr_drop_caches_3
/home/files/sudobins/spr_compact_memory_1
/home/files/sudobins/spr_transparent_hugepage_never
/home/files/sudobins/spr_transparent_hugepage_defrag_never
/home/files/sudobins/spr_transparent_hugepage_always
/home/files/sudobins/spr_transparent_hugepage_defrag_always


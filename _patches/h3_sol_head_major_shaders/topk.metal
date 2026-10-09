// AppleSilicon-Optimized: deterministic fixed-budget video-block routing.
// Global sinks and local neighbors are retained in addition to the budget.
// Included only for the optional top-k mode; Params and tensor ABI are shared.
kernel void sol_route_topk(device const float* QC [[buffer(4)]],
  device const half* KC [[buffer(5)]], device const float2* stats [[buffer(7)]],
  device uchar* routes [[buffer(8)]], constant Params& p [[buffer(21)]],
  device uint* route_bits [[buffer(20)]],
  uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
  uint sg [[simdgroup_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
  constexpr uint SORT_N = SOL_SORT_SIZE;
  const uint J = (SOL_T + 63) / 64, qb = group.x, nh = group.y;
  QC += (ulong(nh) * J + qb) * 128;
  KC += ulong(nh) * ((J + 63) / 64 * 64) * 128;
  routes += (ulong(nh) * J + qb) * J;
  threadgroup float scores[SORT_N];
  threadgroup uint indices[SORT_N];
  const uint sink = min(J, (p.begin + 63) / 64);
  const uint keep = max(1u, uint(ceil(float(J - sink) * SOL_TOPK_RATIO)));
  for (uint j = sg; j < SORT_N; j += 4) {
    float dot = 0;
    if (j < J) for (uint d = lane; d < 128; d += 32)
      dot += QC[d] * float(KC[j * 128 + d]);
    dot = simd_sum(dot);
    if (lane == 0) {
      scores[j] = j >= sink && j < J ? dot * p.scale : -INFINITY;
      indices[j] = j;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Descending bitonic sort, with stable index tie-breaks for equal scores.
  for (uint k = 2; k <= SORT_N; k <<= 1) {
    for (uint stride = k >> 1; stride; stride >>= 1) {
      for (uint i = tid; i < SORT_N; i += 128) {
        const uint other = i ^ stride;
        if (other > i) {
          const bool descending = (i & k) == 0;
          const bool greater = scores[i] > scores[other] ||
            (scores[i] == scores[other] && indices[i] < indices[other]);
          if (greater != descending) {
            const float value = scores[i]; scores[i] = scores[other]; scores[other] = value;
            const uint index = indices[i]; indices[i] = indices[other]; indices[other] = index;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  const bool exact = qb * 64 < p.begin || min((qb + 1) * 64, SOL_T) > p.end;
  // The sorted permutation writes every original route once; no second dot
  // product pass, score cutoff ambiguity, atomics, or device scratch needed.
  for (uint i = tid; i < SORT_N; i += 128) {
    const uint j = indices[i];
    if (j < J) routes[j] = exact || j < sink || abs(int(qb) - int(j)) <= p.local_block_radius || i < keep;
  }
  threadgroup_barrier(mem_flags::mem_device);
  const uint words = (J + 31) / 32;
  for (uint word = tid; word < words; word += 128) {
    uint bits = 0;
    for (uint bit = 0; bit < 32 && word * 32 + bit < J; ++bit)
      bits |= uint(routes[word * 32 + bit] != 0) << bit;
    route_bits[(ulong(nh) * J + qb) * words + word] = bits;
  }
}

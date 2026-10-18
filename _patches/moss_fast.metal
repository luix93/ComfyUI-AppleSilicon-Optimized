// BF16 MOSS operations matching ComfyUI / Comfy Kitchen eager semantics.
// PyTorch evaluates the nonlinear tables; no approximate activation is used.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

inline bfloat rte(float f) {
    uint bits = as_type<uint>(f);
    bits += 0x7fff + ((bits >> 16) & 1);
    return as_type<bfloat>(ushort(bits >> 16));
}

kernel void situ(
    device const bfloat* x, device const bfloat* hidden_table,
    device const bfloat* gate_table, device bfloat* out,
    constant uint& width, constant uint& stride,
    uint i [[thread_position_in_grid]]) {
    uint row = i / width, col = i % width;
    float hidden = float(hidden_table[as_type<ushort>(x[row * stride + col])]);
    float gate = float(gate_table[as_type<ushort>(x[row * stride + width + col])]);
    out[i] = rte(hidden * gate);
}

kernel void rope(
    device const bfloat* q, device const bfloat* k, device const float* pe,
    device bfloat* qo, device bfloat* ko, constant uint& heads,
    constant uint& tokens, uint i [[thread_position_in_grid]]) {
    uint col = i % 128, half_index = col / 64, pair = col % 64;
    uint row = i / 128, token = (row / heads) % tokens;
    uint other = row * 128 + (1 - half_index) * 64 + pair;
    uint f = (token * 64 + pair) * 4 + half_index * 2;
    // Keep both FP32 products and addition separate: FMA changes output bytes.
    float qleft = float(q[i]) * pe[f + half_index];
    float qright = float(q[other]) * pe[f + 1 - half_index];
    float kleft = float(k[i]) * pe[f + half_index];
    float kright = float(k[other]) * pe[f + 1 - half_index];
    qo[i] = rte(qleft + qright);
    ko[i] = rte(kleft + kright);
}

kernel void mod(
    device const bfloat* x, device const bfloat* scale,
    device const bfloat* shift, device bfloat* out,
    constant uint& width, constant uint& tokens,
    uint i [[thread_position_in_grid]]) {
    uint j = (i / (tokens * width)) * width + i % width;
    bfloat s = rte(1.f + float(scale[j]));
    bfloat prod = rte(float(x[i]) * float(s));
    out[i] = rte(float(prod) + float(shift[j]));
}

std::vector<torch::Tensor> fc1_fused(torch::Tensor a, torch::Tensor b, torch::Tensor rs, torch::Tensor cs, c10::optional<torch::Tensor> bias_opt, bool long_schedule, int bm) {
 TORCH_CHECK(a.is_mps() && b.is_mps() && a.is_contiguous() && b.is_contiguous(),"MPS contiguous required");
 TORCH_CHECK(bm==16 || bm==32 || bm==128 || bm==129 || bm==65,"invalid tile");
 uint M=a.size(0), K=a.size(1), F=b.size(0)/2, hb=bias_opt.has_value();
 TORCH_CHECK(F%256==0 && b.size(1)==K,"shape");
 auto opts=a.options().dtype(torch::kBFloat16);
 auto rot=torch::empty({(long)M,(long)F},opts);
 auto gm=torch::empty({(long)M,(long)F/256},a.options().dtype(torch::kFloat));
 auto q=torch::empty({(long)M,(long)F},a.options());
 auto scales=torch::empty({(long)M,1L},a.options().dtype(torch::kFloat));
 auto bias=hb?bias_opt.value().to(torch::kBFloat16).contiguous():torch::zeros({1},opts);
 static id<MTLComputePipelineState> fused[5]={nil},quant=nil;
 int slot=bm==16?0:bm==32?1:bm==128?2:bm==129?3:4;
 if(!fused[slot]) fused[slot]=pso_for_name(bm==16?@"fc1_16":bm==32?@"fc1_32":bm==128?@"fc1_pair128":bm==129?@"fc1_dual128":@"fc1_dual64");
 if(!quant) quant=pso_for_name(@"quant_compact");
 static id<MTLComputePipelineState> rotation=nil; if(bm>=65 && !rotation) rotation=pso_for_name(@"rot_compact");
 int tile=bm==129?128:bm==65?64:bm;
 TORCH_CHECK(bm<65 || K%256==0,"dual K alignment");
 torch::Tensor tensors[9]={a,b,rs,cs,bias,rot,gm,q,scales};
 std::array<id<MTLBuffer>,9> buf;std::array<NSUInteger,9> off;
 for(int i=0;i<9;i++){buf[i]=__builtin_bit_cast(id<MTLBuffer>,tensors[i].storage().data());off[i]=tensors[i].storage_offset()*tensors[i].element_size();}
 auto stream=getCurrentMPSStream();
 dispatch_sync(stream->queue(), ^(){@autoreleasepool{
 auto enc=stream->commandEncoder();[enc setComputePipelineState:fused[slot]];
 for(int i=0;i<7;i++) [enc setBuffer:buf[i] offset:off[i] atIndex:i];
 [enc setBytes:&M length:4 atIndex:7];[enc setBytes:&F length:4 atIndex:8];[enc setBytes:&K length:4 atIndex:9];[enc setBytes:&hb length:4 atIndex:10];
 [enc dispatchThreadgroups:MTLSizeMake(bm>=65?F/128*4:F/256,bm>=65?((M+tile-1)/tile+3)/4:(M+bm-1)/bm,1) threadsPerThreadgroup:MTLSizeMake(bm>=128?512:256,1,1)];
 if(bm>=65) {
 [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];[enc setComputePipelineState:rotation];
 [enc setBuffer:buf[5] offset:off[5] atIndex:0];[enc setBuffer:buf[6] offset:off[6] atIndex:1];
 [enc setBytes:&M length:4 atIndex:2];[enc setBytes:&F length:4 atIndex:3];
 [enc dispatchThreadgroups:MTLSizeMake(M,F/256,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
 }
 [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];[enc setComputePipelineState:quant];
 for(int i=0;i<4;i++) [enc setBuffer:buf[i+5] offset:off[i+5] atIndex:i];
 [enc setBytes:&M length:4 atIndex:4];[enc setBytes:&F length:4 atIndex:5];
 [enc dispatchThreadgroups:MTLSizeMake(M,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
 }});
 return {q,scales};
}

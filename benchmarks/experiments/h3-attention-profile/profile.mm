#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <random>
#include <algorithm>
#include <vector>
int main(int argc,char**argv){@autoreleasepool {
 if(argc<2||argc>3){fprintf(stderr,"usage: runner DIRECTORY [ROUNDS]\n");return 2;}
 NSString*root=@(argv[1]);int reps=argc>2?atoi(argv[2]):8;
 if(reps<1||reps>100){fprintf(stderr,"ROUNDS must be 1..100\n");return 2;}
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();id<MTLCommandQueue>queue=[d newCommandQueue];NSError*err=nil;
 auto load=[&](NSString*name)->id<MTLBuffer>{NSData*x=[NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:name]];if(!x){fprintf(stderr,"missing %s\n",name.UTF8String);exit(2);}return [d newBufferWithBytes:x.bytes length:x.length options:MTLResourceStorageModeShared];};
 id<MTLBuffer>q=load(@"q.bin"),k=load(@"k.bin"),v=load(@"v.bin"),p=load(@"params.bin");
 id<MTLBuffer>o=[d newBufferWithLength:q.length options:MTLResourceStorageModeShared];
 int*pars=(int*)p.contents;int heads=pars[1],n=pars[3];
 NSMutableDictionary*psos=[NSMutableDictionary dictionary];NSMutableDictionary*metadata=[NSMutableDictionary dictionary];
 for(NSString*name in @[@"baseline",@"phases"]){
  NSString*s=[NSString stringWithContentsOfFile:[root stringByAppendingPathComponent:[name stringByAppendingString:@".metal"]] encoding:NSUTF8StringEncoding error:&err];
  MTLCompileOptions*opts=[MTLCompileOptions new];opts.languageVersion=MTLLanguageVersion4_0;opts.mathMode=MTLMathModeSafe;
  id<MTLLibrary>lib=[d newLibraryWithSource:s options:opts error:&err];if(!lib){fprintf(stderr,"%s\n",err.description.UTF8String);return 3;}
  MTLComputePipelineDescriptor*desc=[MTLComputePipelineDescriptor new];desc.computeFunction=[lib newFunctionWithName:@"custom_attn"];
  MTLComputePipelineReflection*reflection=nil;id<MTLComputePipelineState>state=[d newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionBindingInfo reflection:&reflection error:&err];if(!state){fprintf(stderr,"%s\n",err.description.UTF8String);return 4;}
  psos[name]=state;metadata[name]=@{@"max_threads":@(state.maxTotalThreadsPerThreadgroup),@"simd_width":@(state.threadExecutionWidth),@"static_threadgroup_bytes":@(state.staticThreadgroupMemoryLength)};
  id<MTLBinaryArchive>archive=[d newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&err];
  [archive addComputePipelineFunctionsWithDescriptor:desc error:&err];[archive serializeToURL:[NSURL fileURLWithPath:[root stringByAppendingPathComponent:[name stringByAppendingString:@".metalar"]]] error:&err];
 }
 id<MTLCounterSet>timestampSet=nil;
 for(id<MTLCounterSet>set in d.counterSets) if([set.name isEqualToString:MTLCommonCounterSetTimestamp]) timestampSet=set;
 if(!timestampSet||![d supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]){fprintf(stderr,"Stage timestamp sampling unavailable\n");return 5;}
 MTLCounterSampleBufferDescriptor*cd=[MTLCounterSampleBufferDescriptor new];cd.counterSet=timestampSet;cd.storageMode=MTLStorageModeShared;cd.sampleCount=2;
 id<MTLCounterSampleBuffer>counter=[d newCounterSampleBufferWithDescriptor:cd error:&err];if(!counter){fprintf(stderr,"counter: %s\n",err.description.UTF8String);return 5;}
 // Resolved legacy timestamps are nanoseconds on the validated device.
 // Cross-check with the independent command-buffer GPU clock below.
 auto run=[&](int mask,bool save)->NSDictionary*{
  NSString*name=mask<0?@"baseline":@"phases";MTLComputePassDescriptor*pass=[MTLComputePassDescriptor computePassDescriptor];
  pass.sampleBufferAttachments[0].sampleBuffer=counter;pass.sampleBufferAttachments[0].startOfEncoderSampleIndex=0;pass.sampleBufferAttachments[0].endOfEncoderSampleIndex=1;
  id<MTLCommandBuffer>cb=[queue commandBuffer];id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoderWithDescriptor:pass];[enc setComputePipelineState:psos[name]];
  NSArray*buffers=@[q,k,v,o,p,p,p];for(NSUInteger i=0;i<buffers.count;i++)[enc setBuffer:buffers[i] offset:0 atIndex:i];
  uint32_t m=mask;if(mask>=0)[enc setBytes:&m length:4 atIndex:7];
  [enc dispatchThreadgroups:MTLSizeMake(heads,(n+63)/64,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
  if(cb.status==MTLCommandBufferStatusError){fprintf(stderr,"GPU %s\n",cb.error.description.UTF8String);exit(6);}
  NSData*resolved=[counter resolveCounterRange:NSMakeRange(0,2)];const MTLCounterResultTimestamp*t=(const MTLCounterResultTimestamp*)resolved.bytes;
  double ms=(t[1].timestamp-t[0].timestamp)/1e6;
  if(save){NSString*f=mask<0?@"native-baseline.bin":@"native-full-probe.bin";[[NSData dataWithBytesNoCopy:o.contents length:o.length freeWhenDone:NO] writeToFile:[root stringByAppendingPathComponent:f] atomically:NO];}
  return @{@"mask":@(mask),@"counter_ms":@(ms),@"command_buffer_ms":@((cb.GPUEndTime-cb.GPUStartTime)*1000)};
 };
 NSMutableArray*records=[NSMutableArray array];std::vector<int>order={-1,7,6,5,3,4,2,1,0};
 for(int mask:order){NSDictionary*r=run(mask,mask==-1||mask==7);fprintf(stdout,"warm %s\n",r.description.UTF8String);fflush(stdout);}
 std::mt19937 rng(72);
 for(int i=0;i<reps;i++){std::shuffle(order.begin(),order.end(),rng);for(int mask:order){NSMutableDictionary*r=[run(mask,false) mutableCopy];r[@"round"]=@(i);[records addObject:r];}printf("round %d\n",i);fflush(stdout);}
 NSDictionary*report=@{@"device":d.name,@"pipeline_metadata":metadata,@"records":records,@"note":@"Diagnostic phase ablations are not attention implementations; timings are not additive."};
 [[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:nil] writeToFile:[root stringByAppendingPathComponent:@"profile.json"] atomically:YES];
}}

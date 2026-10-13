#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main(int argc,char**argv){@autoreleasepool {
 if(argc!=3){fprintf(stderr,"usage: capture INPUT_DIRECTORY OUTPUT_DIRECTORY\n");return 2;}
 NSString*root=@(argv[1]);NSString*outdir=@(argv[2]);NSError*err=nil;
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();id<MTLCommandQueue>queue=[d newCommandQueue];queue.label=@"H3 exact attention";
 auto load=[&](NSString*name)->id<MTLBuffer>{NSData*x=[NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:name]];if(!x)exit(2);id<MTLBuffer>b=[d newBufferWithBytes:x.bytes length:x.length options:MTLResourceStorageModeShared];b.label=name;return b;};
 id<MTLBuffer>q=load(@"q.bin"),k=load(@"k.bin"),v=load(@"v.bin"),p=load(@"params.bin");
 id<MTLBuffer>o=[d newBufferWithLength:q.length options:MTLResourceStorageModeShared];o.label=@"exact output";
 int*pars=(int*)p.contents;int heads=pars[1],n=pars[3];
 NSString*s=[NSString stringWithContentsOfFile:[root stringByAppendingPathComponent:@"baseline.metal"] encoding:NSUTF8StringEncoding error:&err];
 MTLCompileOptions*opts=[MTLCompileOptions new];opts.languageVersion=MTLLanguageVersion4_0;opts.mathMode=MTLMathModeSafe;
 id<MTLLibrary>lib=[d newLibraryWithSource:s options:opts error:&err];if(!lib){fprintf(stderr,"%s\n",err.description.UTF8String);return 3;}
 id<MTLComputePipelineState>state=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"custom_attn"] error:&err];if(!state){fprintf(stderr,"%s\n",err.description.UTF8String);return 4;}
 auto run=[&](){id<MTLCommandBuffer>cb=[queue commandBuffer];cb.label=@"H3 BQ64 BK32 D128 exact attention";id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];enc.label=@"QK softmax PV";[enc setComputePipelineState:state];NSArray*buffers=@[q,k,v,o,p,p,p];for(NSUInteger i=0;i<buffers.count;i++)[enc setBuffer:buffers[i] offset:0 atIndex:i];[enc dispatchThreadgroups:MTLSizeMake(heads,(n+63)/64,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status==MTLCommandBufferStatusError){fprintf(stderr,"%s\n",cb.error.description.UTF8String);exit(6);}printf("GPU %.3f ms\n",(cb.GPUEndTime-cb.GPUStartTime)*1000);fflush(stdout);};
 for(int i=0;i<3;i++)run();
 MTLCaptureManager*manager=[MTLCaptureManager sharedCaptureManager];MTLCaptureDescriptor*desc=[MTLCaptureDescriptor new];desc.captureObject=queue;desc.destination=MTLCaptureDestinationGPUTraceDocument;desc.outputURL=[NSURL fileURLWithPath:[outdir stringByAppendingPathComponent:@"exact.gputrace"]];
 if(![manager startCaptureWithDescriptor:desc error:&err]){fprintf(stderr,"capture: %s\n",err.description.UTF8String);return 7;}
 run();[manager stopCapture];
 [[NSData dataWithBytesNoCopy:o.contents length:o.length freeWhenDone:NO] writeToFile:[outdir stringByAppendingPathComponent:@"captured-output.bin"] atomically:NO];
 printf("Captured exact attention.\n");
}}

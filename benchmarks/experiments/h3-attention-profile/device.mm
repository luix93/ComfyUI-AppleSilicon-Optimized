#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main(){@autoreleasepool {
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();
 NSMutableArray*sets=[NSMutableArray array];
 for(id<MTLCounterSet>s in d.counterSets){NSMutableArray*c=[NSMutableArray array];for(id<MTLCounter>x in s.counters)[c addObject:x.name];[sets addObject:@{@"name":s.name,@"counters":c}];}
 NSDictionary*r=@{@"timestamp_frequency":@([d queryTimestampFrequency]),@"device":d.name?:@"unavailable",@"counter_sets":sets,@"dispatch_sampling":@([d supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]),@"stage_sampling":@([d supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]),@"recommended_max_working_set":@(d.recommendedMaxWorkingSetSize)};
 NSData*data=[NSJSONSerialization dataWithJSONObject:r options:NSJSONWritingPrettyPrinted error:nil];puts([[NSString alloc]initWithData:data encoding:NSUTF8StringEncoding].UTF8String);
}}

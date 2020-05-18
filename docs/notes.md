# Detecting concurrency

Recorded: 

+ Event before `g_start`
+ Event after `g_stop`
+ Events per stream
    + Before `s_start`
    + After `s_stop`

Times available:
+ `g2g` = `g_start` to `g_stop`
+ Per Stream:
    + `s2s` = `s_start` to `s_stop`
    + `g2s` = `g_start` to `s_stop`
    + `s2g` = `s_start` to `g_stop`


Cases:

+ Sequential - all in 1 stream
+ Streams
    + 4 batches, 4 streams, 1 block uses whole device
    + 4 batches, 4 streams, 1 block uses 1/2 device
    + 4 batches, 4 streams, 1 block uses 1/3 device
    + 4 batches, 4 streams, 1 block uses 1/4 device

Using:

+ GTX 1070
    + 15 SMs
    + 2048 threads per SM
    + blocksize 1024
    + 30 blocks for full utilisiation
    + 1920 cuda cores
    + 

Implementation uses grid stride loop + inner repetition for longer running kernels.


Need to be able to correctly identify

+ full concurrency - 4 batches of 1/4 gpu
+ partial concurrency - multiple streams but not enough hardware 4 batches of 1/2 to 1/3 of a gpu
+ almost no concurrency - multiple streaams but large blocks - some overlap
+ no concurrency - single stream.

## small problem

+ `N = 2 << 10 = 2048`
+ Not enough work to fully use device
+ blocks become idle very quickly, allowing concurrency even when launching enough blocks to use all SMS.
+ Easier to tell when single stream used as time is much longer.

## big problem

+ `N = 2 << 20 = 2097152`
+ This provides more than enough work to fully occupy the device. 
+ profiling suggests full concurrency is achieved *but* total time is approx the same as serial as  just too much work to do.



## small problem with sync

+ `N = 2 << 10 = 2048`
+ Injects device syncs after batches 0 and 2 - a pretend regression.
+ 4 concurrent batches only achives concurrency between batches 1 and 2, but total runtime is approx the same as smaller sizes?



## big problem with sync

+ `N = 2 << 20 = 2097152`
+ Injects device syncs after batches 0 and 2 - a pretend regression.
+ 4 concurrent batches only achives concurrency between batches 1 and 2 so longer runtime than when more of the device is used?
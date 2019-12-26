# Parallelizig K-Means using x86 vector instructions

This code "simd_kmeans.cpp" uses x86 intrinsics in order to parallelize k-means whenever possible. Original serial implementation is in "serial_kmeans.cpp". 
"simd_kmeans.cpp" makes the following improvements over the serial version:
* All calculations are done using row major instead of column major (better cache locality)
* Use 256-bit SIMD instructions for distance and mean calculations
* Passing points by const reference to eliminate copy overhead
* Breaking the dependency in getIDNearestCenter by having an array of distances, one for each cluster and then finding the minimum seperately
* Re-calculating the center of each cluster using 256-bit SIMD instructions

# Evaluation 
Running both implementations (serial and parallel) over the two provided datasets, the parallel version always outperforms the serial one. For the second dataset, speedup up to 8x is achieved. 
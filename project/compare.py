import numpy as np
import pyopencl as cl
import pyopencl.array
import time

# Select the desired OpenCL platform; you shouldn't need to change this:
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()

# Set up a command queue:
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)

gaussian = cl.Program(ctx, """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void gaussian(__global double *a, __global double *b,
                       const uint i) 
{
    uint idx = get_global_id(0);
    uint w = get_global_size(0);    
    double ratio = 0.0;

    if (idx > i) {
        ratio = a[idx * w + i] / a[i * w + i];
        for (int idy = i; idy < w; idy++)
            a[idx * w + idy] -= ratio * a[i * w + idy];

        b[idx] -= ratio * b[i];
    }
}
""").build().gaussian
gaussian.set_scalar_arg_dtypes([None, None, np.uint32])

gaussian_solve = cl.Program(ctx, """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void gaussian_solve(__global double *a, __global double *b, __global double *c,
                             const uint i) 
{
    uint idx = get_global_id(0);
    uint w = get_global_size(0);    

    c[i] = b[i] / a[i * w + i];
    if (idx < i)
        b[idx] -= c[i] * a[idx * w + i];
}
""").build().gaussian_solve
gaussian_solve.set_scalar_arg_dtypes([None, None, None, np.uint32])

gaussian1 = cl.Program(ctx, """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void gaussian1(__global double *a, __global double *b,
                        __local double *a_loc,
                        const uint i) 
{
    uint idx = get_global_id(0);
    uint w = get_global_size(0);    
    double ratio = 0.0;

    for (int idy = i; idy < w; idy++)
        a_loc[idy] = a[i * w + idy];

    if (idx != i) {
        ratio = a[idx * w + i] / a_loc[i];
        for (int idy = i; idy < w; idy++)
            a[idx * w + idy] -= ratio * a_loc[idy];

        b[idx] -= ratio * b[i];
    }
}
""").build().gaussian1
gaussian1.set_scalar_arg_dtypes([None, None, None, np.uint32])

gaussian2 = cl.Program(ctx, """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void gaussian2(__global double *a, __global double *b,
                        __local double *a_loc,
                        const uint i, const uint w) 
{
    uint gid = get_group_id(0);
    uint lid = get_local_id(0);
    double ratio = 0.0;

    // Group Read
    a_loc[lid] = a[i * w + lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid == i || lid < i)
        return;

    ratio = a[gid * w + i] / a_loc[i];

    a[gid * w + lid] -= ratio * a_loc[lid];

    if(lid == i)
        b[gid] -= ratio * b[i];
}
""").build().gaussian2
gaussian2.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32])

gaussian3 = cl.Program(ctx, """
__kernel void gaussian3(__global int *a, __global int *b,
                        __local int *a_loc, const uint w) 
{
    uint idx = get_global_id(0);
    float ratio = 0.0f;

    for (uint i = 0; i < w; i++) {
    	barrier(CLK_LOCAL_MEM_FENCE);
    	for (int idy = i; idy < w; idy++)
        	a_loc[idy] = a[i * w + idy];

    	if (idx == i)
    		continue;

    	ratio = (float)a[idx * w + i] / a_loc[i];

    	for (uint idy = i; idy < w; idy++)
    		atomic_sub(&a[idx * w + idy], (int)(ratio * a_loc[idy]));

    	atomic_sub(&b[idx], (int)(ratio * b[i]));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    b[idx] /= a[idx * w + idx];
}
""").build().gaussian3
gaussian3.set_scalar_arg_dtypes([None, None, None, np.uint32])

def solve1(a, b):
	start = time.time()
	size = len(a)
	res = np.zeros((size, 1))
	for i in xrange(0, size - 1, 1):
		for j in xrange(i + 1, size, 1):
			ratio = a[j][i] / a[i][i]
			b[j] -= ratio * b[i]
			for k in xrange(i, size, 1):
				a[j][k] -= ratio * a[i][k]
	# Solve
	for i in xrange(size - 1, -1, -1):
		res[i] = float(b[i] / a[i][i])
		for j in xrange(i - 1, -1, -1):
			b[j] -= a[j][i] * res[i]
	print 'Naive CPU:	', time.time() - start
	return res

def solve2(a, b):
	size = len(a)
	res = np.zeros(size)
	start = time.time()
	# Upper Triangular Matrix
	for i in xrange(size - 1):
	    ratio = a[i + 1:size, i] / a[i, i]
	    b[i + 1:size] -= ratio * b[i]
	    a[i + 1:size, i:size] -= np.dot(np.reshape(ratio,(size - i - 1, 1)), np.reshape(a[i, i:size], (1,size - i)))

	for i in xrange(size - 1, -1, -1):
		res[i] = float(b[i] / a[i][i])
		b[0:i] = b[0:i] - a[0:i, i].ravel() * res[i]
	print 'Vector CPU:	', time.time() - start
	return res

def solve_gpu_1(a, b):
	a_gpu = cl.array.to_device(queue, a)
	b_gpu = cl.array.to_device(queue, b)
	c_gpu = cl.array.to_device(queue, np.zeros_like(b))
	size = len(a)
	start = time.time()
	for i in xrange(size - 1):
		gaussian(queue, (size,), None, a_gpu.data, b_gpu.data, i)
	for i in xrange(size - 1, -1, -1):
		gaussian_solve(queue, (size,), None, a_gpu.data, b_gpu.data, c_gpu.data, i)

	print 'Optimization1:	', time.time() - start
	res = c_gpu.get()
	return res

def solve_gpu_2(a, b):
	size = len(a)
	a_gpu = cl.array.to_device(queue, a)
	b_gpu = cl.array.to_device(queue, b)
	a_loc = cl.LocalMemory(np.float64().nbytes * size)
	c_gpu = cl.array.to_device(queue, np.zeros_like(b))

	start = time.time()
	for i in xrange(size - 1):
		gaussian1(queue, (size,), None, a_gpu.data, b_gpu.data, a_loc, i)
	for i in xrange(size - 1, -1, -1):
		gaussian_solve(queue, (size,), None, a_gpu.data, b_gpu.data, c_gpu.data, i)

	print 'Optimization2:	', time.time() - start
	res = c_gpu.get()
	return res

def solve_gpu_3(a, b):
	size = len(a)
	a_gpu = cl.array.to_device(queue, a)
	b_gpu = cl.array.to_device(queue, b)
	a_loc = cl.LocalMemory(np.float64().nbytes * size)

	start = time.time()
	for i in xrange(size):
		gaussian2(queue, (size ** 2,), (size,), a_gpu.data, b_gpu.data, a_loc, i, size)
	print 'Optimization3:	', time.time() - start
	res_b = b_gpu.get()
	res_a = a_gpu.get()
	res = res_b / res_a.diagonal()
	return res

def solve_gpu_4(a, b):
	size = len(a)
	a_gpu = cl.array.to_device(queue, np.int32(a))
	b_gpu = cl.array.to_device(queue, np.int32(b))
	a_loc = cl.LocalMemory(np.int32().nbytes * size)

	start = time.time()
	gaussian3(queue, (size,), None, a_gpu.data, b_gpu.data, a_loc, size)
	print 'Optimization4:	', time.time() - start
	res_b = np.float64(b_gpu.get())
	#res_a = np.float64(a_gpu.get())
	#res = res_b / res_a.diagonal()
	return res_b

w = 16

a = np.random.randint(1,255,w**4).reshape(w**2,w**2).astype(np.float64)
aa = np.copy(a)
b = np.random.randint(1,255,w**2).astype(np.float64)
bb = np.copy(b)

res1 = solve1(a, b)

a = np.copy(aa)
b = np.copy(bb)

res2 = solve2(a, b)

a = np.copy(aa)
b = np.copy(bb)

start = time.time()
res3 = np.linalg.solve(a, b)
print 'numpy:	', time.time() - start

a = np.copy(aa)
b = np.copy(bb)

res4 = solve_gpu_1(a, b)

a = np.copy(aa)
b = np.copy(bb)

res5 = solve_gpu_2(a, b)

a = np.copy(aa)
b = np.copy(bb)

res6 = solve_gpu_3(a, b)

a = np.copy(aa)
b = np.copy(bb)

res7 = solve_gpu_4(a, b)

print np.int32(res3 - res7)

print 'cpu_1:	', np.allclose(res3, np.reshape(res1, np.shape(res3)))
print 'cpu_2:	', np.allclose(res3, res2)
print 'gpu_1:	', np.allclose(res3, res4)
print 'gpu_2:	', np.allclose(res3, res5)
print 'gpu_3:	', np.allclose(res3, res6)
print 'gpu_4:	', np.allclose(res3, res7)
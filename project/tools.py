import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy as np
import math
import time
import pyopencl as cl
import pyopencl.array
import os

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
""").build().gaussian

gaussian.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32])

def GE(a, b):
	w = len(b)
	a_gpu = cl.array.to_device(queue, a)
	b_gpu = cl.array.to_device(queue, b)
	a_loc = cl.LocalMemory(np.float64().nbytes * w)

	# Gaussian Elimination
	for i in xrange(w):
		gaussian(queue, (w ** 2,), (w,), a_gpu.data, b_gpu.data, a_loc, i, w)

	a_res = a_gpu.get()
	b_res = b_gpu.get()
	return b_res / a_res.diagonal()

def hist(x):
    bins = np.zeros(256, np.uint32)
    for v in x.flat:
        bins[v] += 1
    return bins

def read_file(path):
	img = mpimg.imread(path)
	return (255.0 / img.max() * (img - img.min())).astype(np.uint8)

def pad(img, BLOCK_SIZE):
	h, w = np.shape(img)
	if h % BLOCK_SIZE != 0:
		img = np.lib.pad(img, [(0, BLOCK_SIZE - h % BLOCK_SIZE), (0, 0)], 'constant', constant_values = 128)

	if w % BLOCK_SIZE != 0:
		img = np.lib.pad(img, [(0, 0), (0, BLOCK_SIZE - w % BLOCK_SIZE)], 'constant', constant_values = 128)

	return img

def reference(img, i_block, j_block, i_max, j_max, BLOCK_SIZE):
	row_ref = [128] * BLOCK_SIZE
	col_ref = [128] * BLOCK_SIZE

	# Dealing with row_ref on the top
	if i_block != 0:
		row_ref = img[i_block * BLOCK_SIZE - 1, j_block * BLOCK_SIZE:(j_block + 1) * BLOCK_SIZE]
	# Dealing with col_ref on the left
	if j_block != 0:
		col_ref = img[i_block * BLOCK_SIZE:(i_block + 1) * BLOCK_SIZE, j_block * BLOCK_SIZE - 1]

	return np.array(col_ref).astype(np.uint8), np.array(row_ref).astype(np.uint8)

def break_block(img, BLOCK_SIZE):
	h, w = np.shape(img)
	i_max = h / BLOCK_SIZE
	j_max = w / BLOCK_SIZE
	block = [0] * (i_max * j_max)
	pad = 0
	for i in xrange(i_max):
		for j in xrange(j_max):
			# Breaking img into blocks
			pad = img[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE, j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE]
			# Pad image blocks with reference points and zeros
			left, top = reference(img, i, j, i_max, j_max, BLOCK_SIZE)

			left = np.insert(np.float64(left), 0, 0, axis = 0)

			top = np.float64(top)

			pad = np.insert(pad, 0, top, axis = 0)
			pad = np.insert(pad, 0, left, axis = 1)

			pad = np.lib.pad(pad, [(0,1), (0,1)], 'constant', constant_values = 0)
			# Group small block, left and top reference points together
			block[i * j_max + j] = pad

	return block

def group_block(block, BLOCK_SIZE, shape):
	# Input block contains only blocked image in a list
	h, w = shape
	i_max = h / BLOCK_SIZE
	j_max = w / BLOCK_SIZE
	img = 0
	for i in xrange(0, i_max):
		tile_base = block[i * j_max]
		for j in xrange(1, j_max):
			tile = block[i * j_max + j]
			tile_base = np.concatenate((tile_base, tile), axis = 1)
		if i == 0:
			img = tile_base
		else:
			img = np.concatenate((img, tile_base), axis = 0)
	return img

def laplacian(mode):
	# DC mode is mode = 1
	# Angular mode is mode = 2-34
	# DC mode Laplacian operator
	delta = np.array([[  0, -32,   0],
					  [-32, 128, -32],
					  [  0, -32,   0]])
	# Displacement Vector
	d = [0, 2, 5, 9, 13, 17, 21, 26, 32]

	if mode >= 2 and mode <= 10:
		delta += np.array([[  0,            0,  0],
						   [-32, 32+d[10-mode], 0],
						   [  0,   -d[10-mode], 0]])
	elif mode >= 11 and mode <= 17:
		delta += np.array([[  0,   -d[mode-10], 0],
						   [-32, 32+d[mode-10], 0],
						   [  0,             0, 0]])
	elif mode >= 18 and mode <= 26:
		delta += np.array([[          0,           -32, 0],
						   [-d[26-mode], 32+d[26-mode], 0],
						   [          0,             0, 0]])
	elif mode >= 27 and mode <= 34:
		delta += np.array([[0,           -32,           0],
						   [0, 32+d[mode-26], -d[mode-26]],
						   [0,             0,           0]])
	elif mode > 34 or mode < 1:
		raise ValueError('invalid angular mode!')

	return np.float64(delta)

def PSNR(ref, block, BLOCK_SIZE):
	ref = np.float64(ref)
	block = np.float64(block)
	dif = np.sum(pow(ref - block, 2)) / (BLOCK_SIZE * BLOCK_SIZE)
	return 20 * np.log10(255.0 / np.sqrt(dif))

def gaussian_elimination(a, b):
	size = len(a)
	res = np.zeros(size)
	for i in xrange(0, size - 1, 1):
		ratio = a[i + 1:size, i] / a[i, i]
		b[i + 1:size] -= ratio * b[i]
		a[i + 1:size, i:size] -= np.dot(np.reshape(ratio,(size - i - 1, 1)), np.reshape(a[i, i:size], (1,size - i)))

	for i in xrange(size - 1, -1, -1):
		res[i] = b[i] / a[i][i]
		b[0:i] = b[0:i] - a[0:i, i] * res[i]
	return res

def inpainting(block, BLOCK_SIZE, processor = 'cpu'):
	
	output = []
	out_mode = []
	block_num = len(block)
	
	time_prepare = 0
	time_solve = 0

	for i in xrange (block_num):
		img = np.float64(block[i])

		psnr_max = 0
		block_max = 0
		mode_max = 0
		
		for mode in xrange(1, 35):
		#for mode in xrange(20,21):

			delta = laplacian(mode)
			delta_right = np.copy(delta)
			delta_right[:,1] += delta_right[:,2]
			delta_bottom = np.copy(delta)
			delta_bottom[1,:] += delta_bottom[2,:]
			delta_last = np.copy(delta_bottom)
			delta_last[:,1] += delta_last[:,2]

			# A, B
			A = np.zeros(BLOCK_SIZE**4).astype(np.float64).reshape(BLOCK_SIZE**2, BLOCK_SIZE**2)
			B = np.zeros(BLOCK_SIZE**2).astype(np.float64)

			start = time.time()

			for j in xrange(1, BLOCK_SIZE + 1):
				if j == BLOCK_SIZE:
					Delta = delta_bottom
				else:
					Delta = delta
				for k in xrange(1, BLOCK_SIZE + 1):
					if k == BLOCK_SIZE:
						if j != BLOCK_SIZE:
							Delta = delta_right
						else:
							Delta = delta_last

					#print j, k
					#print Delta

					result = Delta * img[j-1:j+2, k-1:k+2]

					ind_j = (j - 1) * BLOCK_SIZE + (k - 1)
					# Calculating B[ind_j]
					# Top References
					if j == 1:
						B[ind_j] += -np.sum(result[0, :])
						#print ind_j, j
					
					# Left References
					if k == 1:
						B[ind_j] += -np.sum(result[:, 0])
						#print ind_j, k

					ind_k = ind_j

					# Calculating A[ind_j, :]
					if j == 1 and k == 1:
						A[ind_j, ind_k:ind_k+2] = Delta[1, 1:3]
						ind_k_bottom = ind_k + BLOCK_SIZE
						A[ind_j, ind_k_bottom:ind_k_bottom+2] = Delta[2, 1:3]
					elif j == 1 and k != 1:
						ind_k_bottom = ind_k + BLOCK_SIZE
						if k != BLOCK_SIZE:
							A[ind_j, ind_k-1:ind_k+2] = Delta[1, :]
							A[ind_j, ind_k_bottom-1:ind_k_bottom+2] = Delta[2, :]
						else:
							A[ind_j, ind_k-1:ind_k+1] = Delta[1, 0:2]
							A[ind_j, ind_k_bottom-1:ind_k_bottom+1] = Delta[2, 0:2]
					elif j != 1 and k == 1:
						A[ind_j, ind_k:ind_k+2] = Delta[1, 1:3]
						ind_k_top = ind_k - BLOCK_SIZE
						A[ind_j, ind_k_top:ind_k_top+2] = Delta[0, 1:3]
						if j != BLOCK_SIZE:
							ind_k_bottom = ind_k + BLOCK_SIZE
							A[ind_j, ind_k_bottom:ind_k_bottom+2] = Delta[2, 1:3]
					else:
						ind_k_top = ind_k - BLOCK_SIZE
						ind_k_bottom = ind_k + BLOCK_SIZE
						if j == BLOCK_SIZE and k == BLOCK_SIZE:
							A[ind_j, ind_k-1:ind_k+1] = Delta[1, 0:2]
							A[ind_j, ind_k_top-1:ind_k_top+1] = Delta[0, 0:2]
						elif j != BLOCK_SIZE and k == BLOCK_SIZE:
							A[ind_j, ind_k-1:ind_k+1] = Delta[1, 0:2]
							A[ind_j, ind_k_top-1:ind_k_top+1] = Delta[0, 0:2]
							A[ind_j, ind_k_bottom-1:ind_k_bottom+1] = Delta[2, 0:2]
						elif j == BLOCK_SIZE and k != BLOCK_SIZE:
							A[ind_j, ind_k-1:ind_k+2] = Delta[1, :]
							A[ind_j, ind_k_top-1:ind_k_top+2] = Delta[0, :]
						else:
							A[ind_j, ind_k-1:ind_k+2] = Delta[1, :]
							A[ind_j, ind_k_top-1:ind_k_top+2] = Delta[0, :]
							A[ind_j, ind_k_bottom-1:ind_k_bottom+2] = Delta[2, :]

			#print 'A:	', A[0:BLOCK_SIZE**2,0:BLOCK_SIZE**2]
			#print 'B:	', B[0:BLOCK_SIZE**2]

			time_prepare += time.time() - start
			start = time.time()


			if processor == 'numpy':
				predict = np.linalg.solve(A[0:BLOCK_SIZE**2,0:BLOCK_SIZE**2], B[0:BLOCK_SIZE**2]).reshape(BLOCK_SIZE, BLOCK_SIZE)
			elif processor == 'gpu':
				predict = GE(A, B).reshape(BLOCK_SIZE, BLOCK_SIZE)
			elif processor == 'cpu':
				predict = gaussian_elimination(A, B).reshape(BLOCK_SIZE, BLOCK_SIZE)

			time_solve += time.time() - start

			if mode == 1:
				block_max = predict
				psnr_max = PSNR(img[1:BLOCK_SIZE+1, 1:BLOCK_SIZE+1], block_max, BLOCK_SIZE)
				mode_max = 1

			else:
				psnr = PSNR(img[1:BLOCK_SIZE+1, 1:BLOCK_SIZE+1], predict, BLOCK_SIZE)
				#print psnr
				if psnr > psnr_max:
					psnr_max = psnr
					block_max = predict
					mode_max = mode
		#print block_max
		#print mode_max
		output.append(block_max)
		out_mode.append(mode_max)

		print 'Progress:	', i + 1, '/', block_num

	print 'Solving:	', time_solve
	print 'Prepare:	', time_prepare

	return output, out_mode

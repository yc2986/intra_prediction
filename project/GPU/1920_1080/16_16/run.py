import numpy as np
import tools as tl
import time

bl_size = 16
src = 'source/kimono.png'
dst = 'fig'
processor = 'gpu'

start = time.time()
img = tl.read_file(src)
img = tl.pad(img, bl_size)
shape = np.shape(img)
block = tl.break_block(img, bl_size)
predict, mode_predict = tl.inpainting(block, bl_size, processor = processor)
time_total = time.time() - start
print 'predict size:	', np.shape(predict)
print 'block size:	', np.shape(predict[0])
print 'max:	', np.max(predict)
print 'min:	', np.min(predict)
print time_total
predict = tl.group_block(predict, bl_size, shape)
#predict = np.uint8(predict * (255.0 / np.max(predict)))
#predict = np.uint8(predict)
#plt.imshow(predict, cmap = cm.Greys_r)

np.save(dst, predict)

print 'Done'


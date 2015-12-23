# intra_prediction

HEVC intra prediction based on impainting algorithm

Here is a HEVC intra prediction like tools based on numpy packages accelerated in PyOpenCL.

Linear solver part is accelerate based on lossless and lossy encoding methods. Other parts unfinished.

Another data transfer structure needed to be applied to this to overcome the bottleneck.

python run.py to run the encoding

python compare.py to see the performance difference

python toGrayscale.py to get the encoded image after encoding

Website for this project:
http://gpgpucolumbia2015pvco.weebly.com

import numpy as np
from PIL import Image
from os.path import *
import re

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from struct import unpack
import sys

TAG_CHAR = np.array([202021.25], np.float32)

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data
def readPFM2(file): 
    with open(file, "rb") as f:
            # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

            # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels;
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
        img = np.reshape(img, (height, width))
        img = np.flipud(img)
#        quit()
    return img
    return img, height, width

def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()
def readFlowVKITTI(filename):
    bgr = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    h, w, _c = bgr.shape
    assert bgr.dtype == np.uint16 and _c == 3
            # b == invalid flow flag == 0 for sky or other invalid flow
    invalid = bgr[:,:, 0] == 0
    valid = bgr[:,:,0].astype(np.float32)
                # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 – 1]
    flow = 2.0 / (2**16 - 1.0) * bgr[:,:, 2:0:-1].astype(np.float32) - 1
    temp = flow[:,:,0]
    temp[invalid] = 0
    flow[:,:, 0] = temp * (w - 1)
    temp = flow[:,:,1]
    temp[invalid] = 0
    flow[:,:, 1] = temp * (h - 1)
    return flow, valid



def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid

def readDispKITTI(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    flow = np.stack([-disp, np.zeros_like(disp)], -1)
    return flow, valid
def readDispVKITTI(filename):
    depth = cv2.imread(filename, -1)
    valid = depth.astype(np.float32)
    depth = np.float32(depth)
    disp = (0.532725*725.0087)/(depth / 100.)
    flow = np.stack([-disp, np.zeros_like(disp)], -1)
    return flow, valid



def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])
def readDispCarla(filename, extra_info):
    """ load current file from the list"""
    depth = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
    fov = float(extra_info[0])
    baseline = float(extra_info[1])
    depth = (depth[:, :, 0] + depth[:, :, 1] * 256. + depth[:, :, 2] * 256 * 256.) / (256. * 256 * 256 - 1)
    height, width = np.shape(depth)
    depth = 1000. * depth
    focal = width / (2.0 * np.tan(fov * np.pi / 360))
    depth[depth < 1e-4] = 1e-4
    disp = baseline * focal / depth 
    mn = np.random.randn(height, width)
    tag = mn < 0.5
    tag = np.logical_and(disp < 1e-4, tag)
    disp[tag] = 1000*width
    valid = disp < 2000
    disp = np.stack([-disp, np.zeros_like(disp)], -1)
    return disp, valid
def readDispMiddlebury2(file_name):
    assert basename(file_name) == 'disp0GT.pfm'
    disp = readPFM(file_name).astype(np.float32)
    assert len(disp.shape) == 2
    nocc_pix = file_name.replace('disp0GT.pfm', 'mask0nocc.png')
    assert exists(nocc_pix)
    #nocc_pix = imageio.imread(nocc_pix) == 255
    nocc_pix = np.int32(Image.open(nocc_pix)) == 255
    assert np.any(nocc_pix)
    flow = np.stack([-disp, np.zeros_like(disp)], -1)
    return flow, nocc_pix

def readDispMiddlebury(file_name):
    if basename(file_name) == 'disp0GT.pfm':
        return readDispMiddlebury2(file_name)
    ext = splitext(file_name)[-1]
    if ext == '.pfm':
        #assert basename(file_name) == 'disp0GT.pfm'
        disp = readPFM2(file_name).astype(np.float32)
        assert len(disp.shape) == 2
        #nocc_pix = file_name.replace('disp0GT.pfm', 'mask0nocc.png')
        #assert exists(nocc_pix)
        #nocc_pix = imageio.imread(nocc_pix) == 255
        #assert np.any(nocc_pix)
        flow = np.stack([-disp, np.zeros_like(disp)], -1)
        return flow, (disp > 0) * (disp < 10000)
    elif ext == '.png':
        disp = np.float32(Image.open(file_name))
        flow = np.stack([-disp, np.zeros_like(disp)], -1)
        return flow, (disp > 0) * (disp < 10000)


def readDispPFM(file_name):
    flow = readPFM2(file_name).astype(np.float32)
    if len(flow.shape) == 1:
        flow = np.stack([-flow, np.zeros_like(flow)], -1)
        return flow
    elif len(flow.shape) == 2:
        flow = np.stack([-flow, np.zeros_like(flow)], -1)
        return flow
    else:
        return flow[:, :, :-1]
def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM2(file_name).astype(np.float32)
        if len(flow.shape) == 1:
            flow = np.stack([-flow, np.zeros_like(flow)], -1)
            return flow
        elif len(flow.shape) == 2:
            flow = np.stack([-flow, np.zeros_like(flow)], -1)
            return flow
        else:
            return flow[:, :, :-1]
    return []

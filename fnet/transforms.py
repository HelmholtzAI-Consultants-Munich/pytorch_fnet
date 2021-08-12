import numpy as np
import os
import pdb
import scipy
import warnings
import pdb
import numbers
import random 
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
#from sklearn.preprocessing import OneHotEncoder

def normalize(img): # , min_int=13677, max_int=65535, 
    result = img.astype(np.float64)
    result -= np.mean(result)
    result /= np.std(result)
    return result

class AdaptRange(object):
    def __init__(self, img_min, img_max):
        self.img_min = img_min
        self.img_max = img_max
    def __call__(self, img):
        return (img - self.img_min)/(self.img_max - self.img_min)

class Clip(object):
    def __init__(self, min_clip=None, max_clip=1):
        self.min_clip=min_clip
        self.max_clip=max_clip
    def __call__(self, img):
        return np.clip(img,self.min_clip,self.max_clip)


def threshold(img, min_int=379, max_int=65535):
    (thresh, thresh_img) = cv2.threshold(img, min_int, max_int, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    thresh_img[thresh_img!=0] = 1
    #thresh_img = thresh_img.astype(np.int8)
    return (1.-thresh_img)
    
def do_nothing(img):
    return img.astype(np.float)
'''
class OneHotEncode(object):
    def __init__(self, n_classes=2):
        self.one_hot_encoder = OneHotEncoder() #categories=[n_classes])
    def __call__(self, img):
        self.one_hot_encoder.fit(img)
        return self.one_hot_encoder.transform(img).toarray()
'''
class Propper(object):
    """Padder + Cropper"""
    
    def __init__(self, action='-', **kwargs):
        assert action in ('+', '-')
        
        self.action = action
        if self.action == '+':
            self.transformer = Padder('+', **kwargs)
        else:
            self.transformer = Cropper('-', **kwargs)

    def __repr__(self):
        return 'Propper({})'.format(self.action)

    def __str__(self):
        return '{} => transformer: {}'.format(self.__repr__(), self.transformer)

    def __call__(self, x_in):
        return self.transformer(x_in)

    def undo_last(self, x_in):
        return self.transformer.undo_last(x_in)

class Padder(object):
    def __init__(self, padding='+', by=16, mode='constant'):
        """
        padding: '+', int, sequence
          '+': pad dimensions up to multiple of "by"
          int: pad each dimension by this value
          sequence: pad each dimensions by corresponding value in sequence
        by: int
          for use with '+' padding option
        mode: str
          passed to numpy.pad function
        """
        self.padding = padding
        self.by = by
        self.mode = mode
        
        self.pads = {}
        self.last_pad = None

    def __repr__(self):
        return 'Padder{}'.format((self.padding, self.by, self.mode))

    def _calc_pad_width(self, shape_in):
        if isinstance(self.padding, (str, int)):
            paddings = (self.padding, )*len(shape_in)
        else:
            paddings = self.padding
        pad_width = []
        for i in range(len(shape_in)):
            if isinstance(paddings[i], int):
                pad_width.append((paddings[i],)*2)
            elif paddings[i] == '+':
                padding_total = int(np.ceil(1.*shape_in[i]/self.by)*self.by) - shape_in[i]
                pad_left = padding_total//2
                pad_right = padding_total - pad_left
                pad_width.append((pad_left, pad_right))
        assert len(pad_width) == len(shape_in)
        return pad_width

    def undo_last(self, x_in):
        """Crops input so its dimensions matches dimensions of last input to __call__."""
        assert x_in.shape == self.last_pad['shape_out']
        slices = [slice(a, -b) if (a, b) != (0, 0) else slice(None) for a, b in self.last_pad['pad_width']]
        return x_in[slices].copy()

    def __call__(self, x_in):
        shape_in = x_in.shape
        pad_width = self.pads.get(shape_in, self._calc_pad_width(shape_in))
        x_out = np.pad(x_in, pad_width, mode=self.mode)
        if shape_in not in self.pads:
            self.pads[shape_in] = pad_width
        self.last_pad = {'shape_in': shape_in, 'pad_width': pad_width, 'shape_out': x_out.shape}
        return x_out
    

class Cropper(object):
    def __init__(self, cropping, by=16, offset='mid', n_max_pixels=9732096):
        """Crop input array to given shape."""
        self.cropping = cropping
        self.offset = offset
        self.by = by
        self.n_max_pixels = n_max_pixels
        
        self.crops = {}
        self.last_crop = None

    def __repr__(self):
        return 'Cropper{}'.format((self.cropping, self.by, self.offset, self.n_max_pixels))

    def _adjust_shape_crop(self, shape_crop):
        key = tuple(shape_crop)
        shape_crop_new = list(shape_crop)
        prod_shape = np.prod(shape_crop_new)
        idx_dim_reduce = 0
        order_dim_reduce = list(range(len(shape_crop))[-2:])  # alternate between last two dimensions
        while prod_shape > self.n_max_pixels:
            dim = order_dim_reduce[idx_dim_reduce]
            if not (dim == 0 and shape_crop_new[dim] <= 64):
                shape_crop_new[dim] -= self.by
                prod_shape = np.prod(shape_crop_new)
            idx_dim_reduce += 1
            if idx_dim_reduce >= len(order_dim_reduce):
                idx_dim_reduce = 0
        value = tuple(shape_crop_new)
        print('DEBUG: cropper shape change', shape_crop, 'becomes', value)
        return value

    def _calc_shape_crop(self, shape_in):
        croppings = (self.cropping, )*len(shape_in) if isinstance(self.cropping, (str, int)) else self.cropping
        shape_crop = []
        for i in range(len(shape_in)):
            if croppings[i] is None:
                shape_crop.append(shape_in[i])
            elif isinstance(croppings[i], int):
                shape_crop.append(shape_in[i] - croppings[i])
            elif croppings[i] == '-':
                shape_crop.append(shape_in[i]//self.by*self.by)
            else:
                raise NotImplementedError
        if self.n_max_pixels is not None:
            shape_crop = self._adjust_shape_crop(shape_crop)
        self.crops[shape_in]['shape_crop'] = shape_crop
        return shape_crop

    def _calc_offsets_crop(self, shape_in, shape_crop):
        offsets = (self.offset, )*len(shape_in) if isinstance(self.offset, (str, int)) else self.offset
        offsets_crop = []
        for i in range(len(shape_in)):
            offset = (shape_in[i] - shape_crop[i])//2 if offsets[i] == 'mid' else offsets[i]
            if offset + shape_crop[i] > shape_in[i]:
                warnings.warn('Cannot crop outsize image dimensions ({}:{} for dim {}).'.format(offset, offset + shape_crop[i], i))
                raise AttributeError
            offsets_crop.append(offset)
        self.crops[shape_in]['offsets_crop'] = offsets_crop
        return offsets_crop

    def _calc_slices(self, shape_in):
        shape_crop = self._calc_shape_crop(shape_in)
        offsets_crop = self._calc_offsets_crop(shape_in, shape_crop)
        slices = [slice(offsets_crop[i], offsets_crop[i] + shape_crop[i]) for i in range(len(shape_in))]
        self.crops[shape_in]['slices'] = slices
        return slices

    def __call__(self, x_in):
        shape_in = x_in.shape
        if shape_in in self.crops:
            slices = self.crops[shape_in]['slices']
        else:
            self.crops[shape_in] = {}
            slices = self._calc_slices(shape_in)
        x_out = x_in[slices].copy()
        self.last_crop = {'shape_in': shape_in, 'slices': slices, 'shape_out': x_out.shape}
        return x_out

    def undo_last(self, x_in):
        """Pads input with zeros so its dimensions matches dimensions of last input to __call__."""
        assert x_in.shape == self.last_crop['shape_out']
        shape_out = self.last_crop['shape_in']
        slices = self.last_crop['slices']
        x_out = np.zeros(shape_out, dtype=x_in.dtype)
        x_out[slices] = x_in
        return x_out

    
class Resizer(object):
    def __init__(self, factors):
        """
        factors - tuple of resizing factors for each dimension of the input array"""
        self.factors = factors

    def __call__(self, x):
        #print('Before: ', x.shape)
        x = scipy.ndimage.zoom(x, (self.factors), mode='nearest')
        #print('After: ', x.shape)
        return x

    def __repr__(self):
        return 'Resizer({:s})'.format(str(self.factors)) 

class ReflectionPadder3d(object):
    def __init__(self, padding):
        """Return padded 3D numpy array by mirroring/reflection.

        Parameters:
        padding - (int or tuple) size of the padding. If padding is an int, pad all dimensions by the same value. If
        padding is a tuple, pad the (z, y, z) dimensions by values specified in the tuple."""
        self._padding = None
        
        if isinstance(padding, int):
            self._padding = (padding, )*3
        elif isinstance(padding, tuple):
            self._padding = padding
        if (self._padding == None) or any(i < 0 for i in self._padding):
            raise AttributeError

    def __call__(self, ar):
        return pad_mirror(ar, self._padding)

class Capper(object):
    def __init__(self, low=None, hi=None):
        self._low = low
        self._hi = hi
        
    def __call__(self, ar):
        result = ar.copy()
        if self._hi is not None:
            result[result > self._hi] = self._hi
        if self._low is not None:
            result[result < self._low] = self._low
        return result

    def __repr__(self):
        return 'Capper({}, {})'.format(self._low, self._hi)

    
def pad_mirror(ar, padding):
    """Pad 3d array using mirroring.

    Parameters:
    ar - (numpy.array) array to be padded
    padding - (tuple) per-dimension padding values
    """
    shape = tuple((ar.shape[i] + 2*padding[i]) for i in range(3))
    result = np.zeros(shape, dtype=ar.dtype)
    slices_center = tuple(slice(padding[i], padding[i] + ar.shape[i]) for i in range(3))
    result[slices_center] = ar
    # z-axis, centers
    if padding[0] > 0:
        result[0:padding[0], slices_center[1] , slices_center[2]] = np.flip(ar[0:padding[0], :, :], axis=0)
        result[ar.shape[0] + padding[0]:, slices_center[1] , slices_center[2]] = np.flip(ar[-padding[0]:, :, :], axis=0)
    # y-axis
    result[:, 0:padding[1], :] = np.flip(result[:, padding[1]:2*padding[1], :], axis=1)
    result[:, padding[1] + ar.shape[1]:, :] = np.flip(result[:, ar.shape[1]:ar.shape[1] + padding[1], :], axis=1)
    # x-axis
    result[:, :, 0:padding[2]] = np.flip(result[:, :, padding[2]:2*padding[2]], axis=2)
    result[:, :, padding[2] + ar.shape[2]:] = np.flip(result[:, :, ar.shape[2]:ar.shape[2] + padding[2]], axis=2)
    return result



_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


'''
Code taken on October 19th from : 
https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomAffine
The code was modified so that a list of images can be given and the same random transformations be applied
to all. This is useful in the case of an image and corresponding mask where both need to have the exact same
random transformation applied. Modification applied to __call__ and __init__ functions. 
'''
class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (tuple or int): Optional fill color (Tuple for RGB Image And int for grayscale) for the area
            outside the transform in the output image.(Pillow>=5.0.0)
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and \
                    (len(shear) == 2 or len(shear) == 4), \
                    "shear should be a list or tuple and it must be of length 2 or 4."
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    self.shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    self.shear = [s for s in shear]
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation
        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 2:
                shear = [random.uniform(shears[0], shears[1]), 0.]
            elif len(shears) == 4:
                shear = [random.uniform(shears[0], shears[1]),
                         random.uniform(shears[2], shears[3])]
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img_list):
        """
        Parameters
        ----------
            img_list: List of PIL Images to be transformed
        Returns
        ----------
            img_list: List of images to which the same transformations have been applied
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_list[0].size())
        for idx, image in enumerate(img_list):
            affine_img = TF.affine(image, *ret, resample=self.resample, fillcolor=self.fillcolor)
            img_list[idx] = affine_img
        return img_list

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)
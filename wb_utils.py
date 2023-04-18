import numpy as np
from math import ceil, floor
import torch
from PIL import Image
from sklearn.linear_model import LinearRegression

def get_sobel_kernel(device, chnls=5):
  x_kernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
  x_kernel = torch.tensor(x_kernel, dtype=torch.float32).unsqueeze(0).expand(
    1, chnls, 3, 3).to(device=device)
  x_kernel.requires_grad = False
  y_kernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
  y_kernel = torch.tensor(y_kernel, dtype=torch.float32).unsqueeze(0).expand(
    1, chnls, 3, 3).to(device=device)
  y_kernel.requires_grad = False
  return x_kernel, y_kernel

def aug(img1, img2, img3, img4, img5=None, img6=None):
  if img5 is not None and img6 is not None:
    assert (img1.shape == img2.shape == img3.shape == img4.shape ==
            img5.shape == img6.shape)
  elif img5 is not None:
    assert (img1.shape == img2.shape == img3.shape == img4.shape ==
            img5.shape)
  else:
    assert img1.shape == img2.shape == img3.shape == img4.shape
  aug_op = np.random.randint(4)
  if aug_op == 3:
    scale = np.random.uniform(low=0.75, high=1.25)
  else:
    scale = 1

  h, w, _ = img1.shape
  if aug_op is 1:
    img1 = np.flipud(img1)
    img2 = np.flipud(img2)
    img3 = np.flipud(img3)
    img4 = np.flipud(img4)
    if img5 is not None:
      img5 = np.flipud(img5)
    if img6 is not None:
      img6 = np.flipud(img6)
  elif aug_op is 2:
    img1 = np.fliplr(img1)
    img2 = np.fliplr(img2)
    img3 = np.fliplr(img3)
    img4 = np.fliplr(img4)
    if img5 is not None:
      img5 = np.fliplr(img5)
    if img6 is not None:
      img6 = np.fliplr(img6)
  elif aug_op is 3:
    img1 = imresize(img1, scalar_scale=scale)
    img2 = imresize(img2, scalar_scale=scale)
    img3 = imresize(img3, scalar_scale=scale)
    img4 = imresize(img4, scalar_scale=scale)
    if img5 is not None:
      img5 = imresize(img5, scalar_scale=scale)
    if img6 is not None:
      img6 = imresize(img6, scalar_scale=scale)
  if img5 is not None and img6 is not None:
    return img1, img2, img3, img4, img5, img6
  elif img5 is not None:
    return img1, img2, img3, img4, img5
  else:
    return img1, img2, img3, img4


def extract_patch(img1, img2, img3, img4, img5=None, img6=None,
                  patch_size=256, patch_number=8):
  if img5 is not None and img6 is not None:
    assert (img1.shape == img2.shape == img3.shape == img4.shape ==
            img5.shape == img6.shape)
  elif img5 is not None:
    assert (img1.shape == img2.shape == img3.shape == img4.shape ==
            img5.shape)
  else:
    assert img1.shape == img2.shape == img3.shape == img4.shape
  h, w, c = img1.shape

  # get random patch coord
  for patch in range(patch_number):
    patch_x = np.random.randint(0, high=w - patch_size)
    patch_y = np.random.randint(0, high=h - patch_size)
    if patch == 0:
      patch1 = np.expand_dims(img1[patch_y:patch_y + patch_size,
                              patch_x:patch_x + patch_size, :], axis=0)

      patch2 = np.expand_dims(img2[patch_y:patch_y + patch_size,
                              patch_x:patch_x + patch_size, :], axis=0)

      patch3 = np.expand_dims(img3[patch_y:patch_y + patch_size,
                              patch_x:patch_x + patch_size, :], axis=0)

      patch4 = np.expand_dims(img4[patch_y:patch_y + patch_size,
                              patch_x:patch_x + patch_size, :], axis=0)
      if img5 is not None:
        patch5 = np.expand_dims(img5[patch_y:patch_y + patch_size,
                                patch_x:patch_x + patch_size, :], axis=0)

      if img6 is not None:
        patch6 = np.expand_dims(img6[patch_y:patch_y + patch_size,
                                patch_x:patch_x + patch_size, :], axis=0)

    else:
      patch1 = np.concatenate((patch1, np.expand_dims(
        img1[patch_y:patch_y + patch_size,
        patch_x:patch_x + patch_size, :], axis=0)), axis=0)

      patch2 = np.concatenate((patch2, np.expand_dims(
        img2[patch_y:patch_y + patch_size,
        patch_x:patch_x + patch_size, :], axis=0)), axis=0)

      patch3 = np.concatenate((patch3, np.expand_dims(
        img3[patch_y:patch_y + patch_size,
        patch_x:patch_x + patch_size, :], axis=0)), axis=0)

      patch4 = np.concatenate((patch4, np.expand_dims(
        img4[patch_y:patch_y + patch_size,
        patch_x:patch_x + patch_size, :], axis=0)), axis=0)

      if img5 is not None:
        patch5 = np.concatenate((patch5, np.expand_dims(
          img5[patch_y:patch_y + patch_size,
          patch_x:patch_x + patch_size, :], axis=0)), axis=0)

      if img6 is not None:
        patch6 = np.concatenate((patch6, np.expand_dims(
          img6[patch_y:patch_y + patch_size,
          patch_x:patch_x + patch_size, :], axis=0)), axis=0)

  if img5 is not None and img6 is not None:
    return patch1, patch2, patch3, patch4, patch5, patch6
  elif img5 is not None:
    return patch1, patch2, patch3, patch4, patch5
  else:
    return patch1, patch2, patch3, patch4


def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape

def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale

def triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x>=-1),x<0)
    greaterthanzero = np.logical_and((x<=1),x>=0)
    f = np.multiply((x+1),lessthanzero) + np.multiply((1-x),greaterthanzero)
    return f

def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f

def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length+1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1 # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1) # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices

def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)        
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg =  np.sum(weights*((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg =  np.sum(weights*((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out

def imresize(I, scalar_scale=None, method='bicubic', output_shape=None, mode="vec"):
    if method is 'bicubic':
        kernel = cubic
    elif method is 'bilinear':
        kernel = triangle
    else:
        print ('Error: Unidentified method supplied')
        
    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        print ('Error: scalar_scale OR output_shape should be defined!')
        return
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I) 
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B

def convertDouble2Byte(I):
    B = np.clip(I, 0.0, 1.0)
    B = 255*B
    return np.around(B).astype(np.uint8)

def imread(file, gray=False):
  image = Image.open(file)
  image = np.array(image)
  if not gray:
    image = image[:, :, :3]
  image = im2double(image)
  return image

def im2double(im):
  """ Converts an uint image to floating-point format [0-1].

  Args:
    im: image (uint ndarray); supported input formats are: uint8 or uint16.

  Returns:
    input image in floating-point format [0-1].
  """

  if im[0].dtype == 'uint8' or im[0].dtype == 'int16':
    max_value = 255
  elif im[0].dtype == 'uint16' or im[0].dtype == 'int32':
    max_value = 65535
  return im.astype('float') / max_value

def kernelP(I):
  """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
      Ref: Hong, et al., "A study of digital camera colorimetric characterization
       based on polynomial modeling." Color Research & Application, 2001. """
  return (np.transpose(
    (I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 1], I[:, 0] * I[:, 2],
     I[:, 1] * I[:, 2], I[:, 0] * I[:, 0], I[:, 1] * I[:, 1],
     I[:, 2] * I[:, 2], I[:, 0] * I[:, 1] * I[:, 2],
     np.repeat(1, np.shape(I)[0]))))

def get_mapping_func(image1, image2):
  """ Computes the polynomial mapping """
  image1 = np.reshape(image1, [-1, 3])
  image2 = np.reshape(image2, [-1, 3])
  m = LinearRegression().fit(kernelP(image1), image2)
  return m

def apply_mapping_func(image, m):
  """ Applies the polynomial mapping """
  sz = image.shape
  image = np.reshape(image, [-1, 3])
  result = m.predict(kernelP(image))
  result = np.reshape(result, [sz[0], sz[1], sz[2]])
  return result

def outOfGamutClipping(I):
  """ Clips out-of-gamut pixels. """
  I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
  I[I < 0] = 0  # any pixel is below 0, clip it to 0
  return I

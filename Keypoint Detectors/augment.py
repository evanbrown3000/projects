import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import tensorflow as tf

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

def simple_preprocess_fn(x , y ):
    x = tf.reshape(x, shape=(1,*x.shape))
    y = tf.reshape(y, shape=(1,*y.shape))
    seq = iaa.Sequential(
        [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(translate_px={"x": (50, 50)}, rotate=(-10, 10)),
        ],
        random_order=True
        )
    x,y = seq(images=x.numpy(), keypoints=y.numpy()) 
    x = x.reshape(x.shape[1:])
    y = y.reshape(y.shape[1:])  
    return (x,y)

def preprocess_fn(x , y ):
      x = tf.reshape(x, shape=(1,*x.shape))
      y = tf.reshape(y, shape=(1,*y.shape))
      seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        iaa.Affine(translate_percent={"x": (-.4, .4), "y": (-.4, .4)}),# translate by -n to +n percent (per axis)
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
        ))
    ],
        random_order=True
      )
      # seq = tf.py_function(func=seq, inp=[x,y], Tout=tf.float32)
      x,y = seq(images=x.numpy(), keypoints=y.numpy()) 

      x = x.reshape(x.shape[1:])
      y = y.reshape(y.shape[1:])


      #TESTING````````````````````````````

      #     print(x.shape)        
      #     print(y.shape, y)  

      #     plt.imshow(x/255.)
      #     plt.scatter(y[:, 0], y[:, 1], s=20, marker='.', c='m') 
      #     plt.show()        
      return (x,y)


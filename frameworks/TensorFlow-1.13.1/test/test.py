import tensorflow as tf
import numpy as np
# 
#  load graph
f = open('frozen.pb', 'rb')
gd = tf.GraphDef.FromString(f.read())
tf.import_graph_def(gd, name='')

# prepare input image ([-1,1), NHWC)
# NOTE: PIL.resize() has [W,H] as param, but the shape of the result is [H,W,C]
# test_image = np.array([np.array(PIL.Image.open(INPUT_IMAGE).resize([480,270])).astype(np.float)/128-1])
test_image = np.zeros((1,128,128,3))

# eval the endpoints
g = tf.get_default_graph()
with tf.Session() as sess:
    test = sess.run('MobilenetV1/Conv_13_pointwise/act_quant/FakeQuantWithMinMaxVars:0', feed_dict={'images:0': test_image})

print(test.shape)
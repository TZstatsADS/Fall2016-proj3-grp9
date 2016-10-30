###### TEST IMAGES 
image = caffe.io.load_image(caffe_root + 'examples/images/chicken_0046.jpg')
plt.imshow(image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformer.preprocess('data', image)
net.forward()
feature = np.array(net.blobs['conv1'].data[0])

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')

net.blobs['fc6'].data[0].shape

filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))


a = np.reshape(net.blobs['conv1'].data[0], 290400, order='C')
#a = np.reshape(net.params['conv1'].data[0], 290400, order='C')
a.max()

feat = net.blobs['norm1'].data[0]
vis_square(feat)



import pylab
pylab.show()

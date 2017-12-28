def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: Tensor of shape (1, H, W, 3) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!
    iShape = tf.shape(img)   
    i_result = img[:, 1:, :, :] - img[:, :-1, :, :]
    j_result = img[:, :, 1:, :] - img[:, :, :-1, :]
    loss = tv_weight * (tf.reduce_sum(tf.square(i_result)) + tf.reduce_sum(tf.square(j_result)))
    return loss

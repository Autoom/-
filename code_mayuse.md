## keras recall and precision 
### Keras Recall and Precision

    
	def precision(y_true, y_pred):
	    """Precision metric.
	    Only computes a batch-wise average of precision.
	    Computes the precision, a metric for multi-label classification of
	    how many selected items are relevant.
	    """
	    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	    precision = true_positives / (predicted_positives + K.epsilon())
	    return precision

	def recall(y_true, y_pred):
	    """Recall metric.
	    Only computes a batch-wise average of recall.
	    Computes the recall, a metric for multi-label classification of
	    how many relevant items are selected.
	    """
	    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	    recall = true_positives / (possible_positives + K.epsilon())
	    return recall
we can use them in 

    model.compile(...,metrics=['acc',recall,precision])
    
### Keras custom decision threshold for precision and recall
> keras 中的recall 和 precision 的阈值设置为0.5， 在这里的阈值可以随意设置

    def precision_threshold(threshold=0.5):
	    def precision(y_true, y_pred):
	        """Precision metric.
	        Computes the precision over the whole batch using threshold_value.
	        """
	        threshold_value = threshold
	        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
	        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
	        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
	        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
	        # count the predicted positives
	        predicted_positives = K.sum(y_pred)
	        # Get the precision ratio
	        precision_ratio = true_positives / (predicted_positives + K.epsilon())
	        return precision_ratio
	    return precision

	def recall_threshold(threshold = 0.5):
	    def recall(y_true, y_pred):
	        """Recall metric.
	        Computes the recall over the whole batch using threshold_value.
	        """
	        threshold_value = threshold
	        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
	        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
	        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
	        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
	        # Compute the number of positive targets.
	        possible_positives = K.sum(K.clip(y_true, 0, 1))
	        recall_ratio = true_positives / (possible_positives + K.epsilon())
	        return recall_ratio
	    return recall
we can use them in 

    model.compile(..., metrics = [precision_threshold(0.1), precision_threshold(0.2),precision_threshold(0.8), recall_threshold(0.2,...)])
    
    
    
## Loss Function
### focal loss Keras
> 解决多分类的样本不平衡问题

    from keras import backend as K
	'''
	Compatible with tensorflow backend
	'''
	def focal_loss(gamma=2., alpha=.25):
		def focal_loss_fixed(y_true, y_pred):
			pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
	        	pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
	        	return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
		return focal_loss_fixed
we can use them in 

    model.compile(optimizer=optimizer, loss=[focal_loss(alpha=.25, gamma=2)])

# tensorflow 数据增强
> 通常用得到的数据增强函数

	def parse_data(filename, scores):
	    image = tf.read_file(filename)
	    image = tf.image.decode_jpeg(image, channels=3)
	    image = tf.image.resize_images(image, (IMAGE_SIZE, IMAGE_SIZE))
	    image = tf.image.random_flip_left_right(image) #左右翻转
	    image = tf.contrib.image.rotate(image, random.randint(-30,30) * math.pi/180,interpolation='BILINEAR')# 图像旋转（按照中心点旋转(-30,30)度）
	    image = tf.image.random_hue(image, max_delta=0.05) # 色度
	    image = tf.image.random_contrast(image, lower=0.3, upper=1.0) # 对比度
	    image = tf.image.random_brightness(image, max_delta=0.2) # 亮度
	    image = tf.image.random_saturation(image, lower=0.5, upper=2.0) #饱和度
	    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
	    return image, scores
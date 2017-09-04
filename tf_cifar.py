
import tensorflow as tf
from tqdm import tqdm
import load_cifar

batch_size = 256


# trainset, testset = load_cifar.load()


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var
def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var
def make_conv(name, in_tensor, in_filter_num, out_filter_num, size=3, wd=1e-5):
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('weights',
                                            shape=[size, size, in_filter_num, out_filter_num],
                                            stddev=5e-2,
                                            wd=wd)
        conv = tf.nn.conv2d(in_tensor, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [out_filter_num], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        bn = tf.nn.batch_normalization(pre_activation)
        conv1 = tf.nn.relu(bn, name=scope.name)
        return conv1
def inference(images):
    """Build the CIFAR-10 model.
    Args:
        images: Images returned from distorted_inputs() or inputs().
    Returns:
        Logits.
    """

    conv11 = make_conv("conv11", images, 3, 64)
    conv12 = make_conv("conv12", conv11, 64, 64)
    pool1 = tf.nn.max_pool(conv12, ksize=[1, 1, 2, 2], strides=[1, 1, 1, 1],
                          name='pool1')

    conv21 = make_conv("conv21", pool1,  64,  128)
    conv22 = make_conv("conv22", conv21, 128, 128)
    pool2 = tf.nn.max_pool(conv22, ksize=[1, 1, 2, 2], strides=[1, 1, 1, 1],
                          name='pool2')

    conv31 = make_conv("conv31", pool2,  128, 256)
    conv32 = make_conv("conv32", conv31, 256, 256)
    conv33 = make_conv("conv33", conv32, 256, 256)
    conv34 = make_conv("conv34", conv33, 256, 256)
    pool3 = tf.nn.max_pool(conv34, ksize=[1, 1, 2, 2], strides=[1, 1, 1, 1],
                          name='pool3')

    conv41 = make_conv("conv41", pool3,  256, 512)
    conv42 = make_conv("conv42", conv41, 512, 512)
    conv43 = make_conv("conv43", conv42, 512, 512)
    conv44 = make_conv("conv44", conv43, 512, 512)
    pool4 = tf.nn.max_pool(conv44, ksize=[1, 1, 2, 2], strides=[1, 1, 1, 1],
                          name='pool4')

    conv51 = make_conv("conv51", pool4,  512, 512)
    conv52 = make_conv("conv52", conv51, 512, 512)
    conv53 = make_conv("conv53", conv52, 512, 512)
    conv54 = make_conv("conv54", conv53, 512, 512)
    pool5 = tf.nn.max_pool(conv54, ksize=[1, 1, 2, 2], strides=[1, 1, 1, 1],
                          name='pool5')

    avgpool = tf.nn.avg_pool(pool5, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
                             name="avgpool")
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [512, NUM_CLASSES],
                                            stddev=1/512.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(avgpool, weights), biases, name=scope.name)

    return softmax_linear

def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]
    Returns:
        Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def train(total_loss, global_step):
    """Train CIFAR-10 model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
        train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

initial_lr = 0.01
trainer = gluon.Trainer(vgg_net.collect_params(), 'sgd',
                {'learning_rate': initial_lr, 'momentum':0.9, 'wd':5e-4})
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
#evaluate_accuracy(test_data,vgg_net)
###########################
#  Only one epoch so tests can run quickly, increase this variable to actually run
###########################

import time

initial_lr = 0.01
trainer = gluon.Trainer(vgg_net.collect_params(), 'sgd', {'learning_rate': initial_lr})
best_acc=0
start_epoch=0
timer = 0
for epoch in range(start_epoch,10):  # loop over the dataset multiple times
    start=time.time()
    train_loss = 0.0
    for i, (d, l) in enumerate(tqdm(train_data)):
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        with autograd.record():
            output = vgg_net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])

        ##########################
        #  Keep a moving average of the losses
        ##########################
        train_loss += nd.mean(loss).asscalar()

    test_loss, test_accuracy = evaluate_accuracy(test_data, vgg_net)
    train_loss = train_loss / len(train_data)
    batch_time = (time.time()-start)/len(train_data)
    timer += time.time()-start
    print('Epoch %3d: time: %5.2fs, loss: %5.4f, test loss: %5.4f, accuracy : %5.2f %%, %.3fs/epoch, %.3fms/batch' %
          (epoch, timer, train_loss, test_loss, test_accuracy*100, time.time()-start, batch_time))

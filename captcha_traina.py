from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime
import argparse
import sys

import tensorflow as tf
import captcha_modela as captcha

FLAGS = None

def run_train():
  """Train CAPTCHA for a number of steps."""

  with tf.Graph().as_default():
    images, labels = captcha.inputs(train=True, batch_size=FLAGS.batch_size)
#    test_images, test_labels = captcha.inputs(train=False, batch_size=FLAGS.batch_size)

    logits = captcha.inference(images, keep_prob=0.7, is_training=True)
#    test_logits = captcha.inference(test_images, keep_prob=1, is_training=False)

    loss = captcha.loss(logits, labels)
    correct = captcha.evaluation(logits, labels)
#    test_correct = captcha.evaluation(test_logits, test_labels)
    eval_correct = correct/FLAGS.batch_size
    
    tf.summary.scalar('precision', eval_correct)
    tf.summary.scalar('loss', loss)
    tf.summary.image('image', images, 10)
    summary = tf.summary.merge_all()
    
    train_op = captcha.training(loss)

    saver = tf.train.Saver(tf.global_variables())

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        
        sess.run(init_op)
        
        summary_writer = tf.summary.FileWriter( FLAGS.train_dir, sess.graph)
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
          step = 0
          while not coord.should_stop():
            start_time = time.time()
            _, loss_value, train_co = sess.run([train_op, loss, correct])
            result = sess.run(summary) #merged也是需要run的  
            summary_writer.add_summary(result, step)  
            summary_writer.flush()
            
            duration = time.time() - start_time
            if step % 100 == 0:
              print('>> Step %d run_train: loss = %.2f  (%.3f sec)' %
                                                         (step, loss_value, duration))
#            with open('test.txt', 'a') as f:
#                f.write(str(test_co)+'\n')
            if step % 300 == 0:
              print('>> %s Saving in %s' % (datetime.now(), FLAGS.checkpoint))
              saver.save(sess, FLAGS.checkpoint, global_step=step)
            step += 1
            if step>200000:
               break
        except Exception as e:
          print('>> %s Saving in %s' % (datetime.now(), FLAGS.checkpoint))
          saver.save(sess, FLAGS.checkpoint, global_step=step)
          coord.request_stop(e)
        finally:
          coord.request_stop()
        coord.join(threads)


def main(_):
#  if tf.gfile.Exists(FLAGS.train_dir):
#    tf.gfile.DeleteRecursively(FLAGS.train_dir)  # 删除路径下的所有文件
#  tf.gfile.MakeDirs(FLAGS.train_dir)
  run_train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--batch_size',
      type=int,
      default=32,
      help='Batch size.'
  )
  parser.add_argument(
      '--train_dir',
      type=str,
      default='./captcha_traina',
      help='Directory where to write event logs.'
  )
  parser.add_argument(
      '--checkpoint',
      type=str,
      default='./captcha_traina/captcha',
      help='Directory where to write checkpoint.'
  )
  FLAGS, unparsed = parser.parse_known_args()
#  main()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)





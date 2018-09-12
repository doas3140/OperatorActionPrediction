
import tensorflow as tf
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
from tensorboard import summary as summary_lib


class TensorBoardWriter(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)
    
    def __enter__(self):
        return self
    
    def __exit__(self, unused_type, unused_value, unused_traceback):
        self.writer.close()

    def log_scalar(self, value, name, global_step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=name,
                                                     simple_value=value)])
        self.writer.add_summary(summary, global_step)

    def log_histogram(self, values, name, global_step, bins=100):
        values = np.array(values)
        counts, bin_edges = np.histogram(values, bins=bins)
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))
        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, histo=hist)])
        self.writer.add_summary(summary, global_step)
        self.writer.flush()

    def add_summary(self,summary):
        self.writer.add_summary(summary)

    def log_pr_curve(self,tp,fp,tn,fn,name,global_step,display_name='No display_name',description='No description'):
        recall = np.clip((tp / (tp + fn)),0,1)
        precision = np.clip((tp / (tp + fp)),0,1)
        num_thresholds = len(precision)
        summary_proto = summary_lib.pr_curve_raw_data_pb(
                            name=name,
                            true_positive_counts=tp,
                            false_positive_counts=fp,
                            true_negative_counts=tn,
                            false_negative_counts=fn,
                            precision=precision,
                            recall=recall,
                            num_thresholds=num_thresholds,
                            display_name=display_name,
                            description=description
                        )
        self.writer.add_summary(summary_proto,global_step)
        self.writer.flush()

    # def log_images(self, tag, images, global_step):
    #     im_summaries = []
    #     for nr, img in enumerate(images):
    #         s = StringIO()
    #         plt.imsave(s, img, format='png')
    #         img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
    #                                    height=img.shape[0],
    #                                    width=img.shape[1])
    #         im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
    #                                              image=img_sum))
    #     summary = tf.Summary(value=im_summaries)
    #     self.writer.add_summary(summary, global_step)
# tensorboard components
import tensorflow as tf
import os

################################################################################
# trace graph
################################################################################
class GraphTracer:
    def __init__(self, logdir=None):
        self.logdir = logdir
        self.writer = tf.summary.create_file_writer(logdir)
        self.is_tracing = False
        self.first_call = True
        print('trace will be saved to:', logdir)

    def on_batch_begin(self):
        if self.first_call:
            tf.summary.trace_on(graph=True, profiler=True)
            self.is_tracing = True
            self.first_call = False

    def on_batch_end(self):
        if self.is_tracing:
            with self.writer.as_default():
                tf.summary.trace_export(
                    name="graph_trace",
                    step=0,
                    profiler_outdir=os.path.join(self.logdir, 'train'))
            self.is_tracing = False


"""
Sentence Pair Classification with Bidirectional Encoder Representations from Transformers

=========================================================================================

This example shows how to implement finetune a model with pre-trained BERT parameters for
sentence pair classification, with Gluon NLP Toolkit.

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
"""

# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation

import io
import os
import time
import argparse
import random
import logging
import warnings
import multiprocessing
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
from gluonnlp.model import get_bert_model
from gluonnlp.data import BERTTokenizer
from quantization import *

from bert import BERTClassifier, BERTRegression
from dataset import MRPCTask, QQPTask, RTETask, STSBTask, \
    QNLITask, CoLATask, MNLITask, WNLITask, SSTTask, BERTDatasetTransform

tasks = {
    'MRPC': MRPCTask(),
    'QQP': QQPTask(),
    'QNLI': QNLITask(),
    'RTE': RTETask(),
    'STS-B': STSBTask(),
    'CoLA': CoLATask(),
    'MNLI': MNLITask(),
    'WNLI': WNLITask(),
    'SST': SSTTask()
}

parser = argparse.ArgumentParser(
    description='BERT fine-tune examples for GLUE tasks.')
parser.add_argument(
    '--epochs', type=int, default=0, help='number of epochs, default is 3')
parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='Batch size. Number of examples per gpu in a minibatch, default is 32')
parser.add_argument(
    '--dev_batch_size',
    type=int,
    default=8,
    help='Batch size for dev set and test set, default is 8')
parser.add_argument(
    '--optimizer',
    type=str,
    default='bertadam',
    help='Optimization algorithm, default is bertadam')
parser.add_argument(
    '--lr',
    type=float,
    default=5e-5,
    help='Initial learning rate, default is 5e-5')
parser.add_argument(
    '--epsilon',
    type=float,
    default=1e-06,
    help='Small value to avoid division by 0, default is 1e-06'
)
parser.add_argument(
    '--warmup_ratio',
    type=float,
    default=0.1,
    help='ratio of warmup steps used in NOAM\'s stepsize schedule, default is 0.1')
parser.add_argument(
    '--log_interval',
    type=int,
    default=10,
    help='report interval, default is 10')
parser.add_argument(
    '--max_len',
    type=int,
    default=128,
    help='Maximum length of the sentence pairs, default is 128')
parser.add_argument(
    '--pad',
    action='store_true',
    help='Whether to pad to maximum length when preparing data batches. Default is False.')
parser.add_argument(
    '--seed', type=int, default=2, help='Random seed, default is 2')
parser.add_argument(
    '--accumulate',
    type=int,
    default=None,
    help='The number of batches for gradients accumulation to simulate large batch size. '
         'Default is None')
parser.add_argument(
    '--gpu', type=int, default=None, help='Which gpu for finetuning. By default cpu is used.')
parser.add_argument(
    '--task_name',
    type=str,
    choices=tasks.keys(),
    help='The name of the task to fine-tune. Choices include MRPC, QQP, '
         'QNLI, RTE, STS-B, CoLA, MNLI, WNLI, SST.')
parser.add_argument(
    '--bert_model',
    type=str,
    default='bert_12_768_12',
    help='The name of pre-trained BERT model to fine-tune'
    '(bert_24_1024_16 and bert_12_768_12).')
parser.add_argument(
    '--bert_dataset',
    type=str,
    default='book_corpus_wiki_en_uncased',
    help='The dataset BERT pre-trained with.'
    'Options include \'book_corpus_wiki_en_cased\', \'book_corpus_wiki_en_uncased\''
    'for both bert_24_1024_16 and bert_12_768_12.'
    '\'wiki_cn_cased\', \'wiki_multilingual_uncased\' and \'wiki_multilingual_cased\''
    'for bert_12_768_12 only.')
parser.add_argument(
    '--pretrained_bert_parameters',
    type=str,
    default=None,
    help='Pre-trained bert model parameter file. default is None')
parser.add_argument(
    '--model_parameters',
    type=str,
    default=None,
    help='A parameter file for the model that is loaded into the model'
    ' before training/inference. It is different from the parameter'
    ' file written after the model is trained. default is None')
parser.add_argument(
    '--output_dir',
    type=str,
    default='./output_dir',
    help='The output directory where the model params will be written.'
    ' default is ./output_dir')
parser.add_argument(
    '--only-calibration',
    action='store_true',
    help='If set, we skip training and only perform inference on dev and test data.')
parser.add_argument('--calib-mode', type=str, default='naive',
                    help='calibration mode used for generating calibration table for the quantized symbol; supports'
                         ' 1. none: no calibration will be used. The thresholds for quantization will be calculated'
                         ' on the fly. This will result in inference speed slowdown and loss of accuracy'
                         ' in general.'
                         ' 2. naive: simply take min and max values of layer outputs as thresholds for'
                         ' quantization. In general, the inference accuracy worsens with more examples used in'
                         ' calibration. It is recommended to use `entropy` mode as it produces more accurate'
                         ' inference results.'
                         ' 3. entropy: calculate KL divergence of the fp32 output and quantized output for optimal'
                         ' thresholds. This mode is expected to produce the best inference accuracy of all three'
                         ' kinds of quantized models if the calibration dataset is representative enough of the'
                         ' inference dataset.')

parser.add_argument('--num-calib-batches',
                    type=int,
                    default=5,
                    help='batch number for calibration')

parser.add_argument('--quantized-dtype', type=str, default='auto',
                    choices=['int8', 'uint8', 'auto'],
                    help='quantization destination data type for input data')

parser.add_argument('--enable-calib-quantize', type=bool, default=True,
                    help='If enabled, the quantize op will '
                         'be calibrated offline if calibration mode is '
                         'enabled')

args = parser.parse_args()

logging.getLogger().setLevel(logging.DEBUG)
logging.captureWarnings(True)
logging.info(args)

max_len =args.max_len
batch_size = args.batch_size
dev_batch_size = args.dev_batch_size
task_name = args.task_name
accumulate = args.accumulate
pad = args.pad
only_calibration = args.only_calibration
log_interval = args.log_interval * accumulate if accumulate else args.log_interval
if accumulate:
    logging.info('Using gradient accumulation. Effective batch size = %d',
                 accumulate * batch_size)

# random seed
np.random.seed(args.seed)
random.seed(args.seed)
mx.random.seed(args.seed)

ctx = mx.cpu() if args.gpu is None else mx.gpu(args.gpu)

task = tasks[task_name]

# model and loss
model_name = args.bert_model
dataset = args.bert_dataset
epochs = args.epochs
model_parameters = args.model_parameters
# calibration
num_calib_batches = args.num_calib_batches
calib_mode = args.calib_mode
quantized_dtype = args.quantized_dtype

get_pretrained = False
bert, vocabulary = get_bert_model(
    model_name=model_name,
    dataset_name=dataset,
    pretrained=get_pretrained,
    ctx=ctx,
    use_pooler=True,
    use_decoder=False,
    use_classifier=False)

# if not task.class_labels:
#     # STS-B is a regression task.
#     # STSBTask().class_labels returns None
#     model = BERTRegression(bert, dropout=0.1)
#     if not model_parameters:
#         model.regression.initialize(init=mx.init.Normal(0.02), ctx=ctx)
#     loss_function = gluon.loss.L2Loss()
# else:
#     model = BERTClassifier(
#         bert, dropout=0.1, num_classes=len(task.class_labels))
#     if not model_parameters:
#         model.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
#     loss_function = gluon.loss.SoftmaxCELoss()

# # load checkpointing
# output_dir = args.output_dir
# if pretrained_bert_parameters:
#     logging.info('loading bert params from %s', pretrained_bert_parameters)
#     model.bert.load_parameters(pretrained_bert_parameters, ctx=ctx,
#                                ignore_extra=True)
# if model_parameters:
#     logging.info('loading model params from %s', model_parameters)
#     model.load_parameters(model_parameters, ctx=ctx)
# nlp.utils.mkdir(output_dir)

# logging.info(model)
# model.hybridize(static_alloc=True)

network, args, auxs = mx.model.load_checkpoint(model_parameters, epochs)
data_shape0 = (dev_batch_size, max_len)
data_shape1 = (dev_batch_size, max_len)
data_shape2 = (dev_batch_size,)
data_names = ('data0', 'data1', 'data2')
if ctx == mx.cpu():
    network = network.get_backend_symbol('MKLDNN')
mod = mx.mod.Module(symbol=network,  context=ctx, data_names=data_names, label_names=None)
mod.bind(for_training=False,
            data_shapes=[('data0', data_shape0), ('data1', data_shape1), ('data2', data_shape2)])
mod.set_params(args, auxs, allow_missing=False, force_init=True)
loss_function = gluon.loss.SoftmaxCELoss()
loss_function.hybridize(static_alloc=True)

# data processing
do_lower_case = 'uncased' in dataset
bert_tokenizer = BERTTokenizer(vocabulary, lower=do_lower_case)

def preprocess_data(tokenizer, task, batch_size, dev_batch_size, max_len, pad=False):
    """Train/eval Data preparation function."""
    pool = multiprocessing.Pool()

    # transformation for data train and dev
    label_dtype = 'float32' if not task.class_labels else 'int32'
    trans = BERTDatasetTransform(tokenizer, max_len,
                                 class_labels=task.class_labels,
                                 pad=pad, pair=task.is_pair,
                                 has_label=True)

    # data train
    # task.dataset_train returns (segment_name, dataset)
    train_tsv = task.dataset_train()[1]
    data_train = mx.gluon.data.SimpleDataset(pool.map(trans, train_tsv))
    data_train_len = data_train.transform(
        lambda input_id, length, segment_id, label_id: length, lazy=False)
    # bucket sampler for training
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
        nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(label_dtype))
    batch_sampler = nlp.data.sampler.FixedBucketSampler(
        data_train_len,
        batch_size=batch_size,
        num_buckets=10,
        ratio=0,
        shuffle=True)
    # data loader for training
    loader_train = gluon.data.DataLoader(
        dataset=data_train,
        num_workers=1,
        batch_sampler=batch_sampler,
        batchify_fn=batchify_fn)

    # data dev. For MNLI, more than one dev set is available
    dev_tsv = task.dataset_dev()
    dev_tsv_list = dev_tsv if isinstance(dev_tsv, list) else [dev_tsv]
    loader_dev_list = []
    for segment, data in dev_tsv_list:
        data_dev = mx.gluon.data.SimpleDataset(pool.map(trans, data))
        loader_dev = mx.gluon.data.DataLoader(
            data_dev,
            batch_size=dev_batch_size,
            num_workers=1,
            shuffle=False,
            batchify_fn=batchify_fn)
        loader_dev_list.append((segment, loader_dev))

    # batchify for data test
    test_batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
        nlp.data.batchify.Pad(axis=0))
    # transform for data test
    test_trans = BERTDatasetTransform(tokenizer, max_len,
                                      class_labels=None,
                                      pad=pad, pair=task.is_pair,
                                      has_label=False)

    # data test. For MNLI, more than one test set is available
    test_tsv = task.dataset_test()
    test_tsv_list = test_tsv if isinstance(test_tsv, list) else [test_tsv]
    loader_test_list = []
    for segment, data in test_tsv_list:
        data_test = mx.gluon.data.SimpleDataset(pool.map(test_trans, data))
        loader_test = mx.gluon.data.DataLoader(
            data_test,
            batch_size=dev_batch_size,
            num_workers=1,
            shuffle=False,
            batchify_fn=test_batchify_fn)
        loader_test_list.append((segment, loader_test))
    return loader_train, loader_dev_list, loader_test_list, len(data_train)


# Get the loader.
logging.info('processing dataset...')
train_data, dev_data_list, test_data_list, num_train_examples = preprocess_data(
    bert_tokenizer, task, batch_size, dev_batch_size, max_len, pad)


def log_eval(batch_id, batch_num, metric, step_loss, log_interval):
    """Generate and print out the log message for inference. """
    metric_nm, metric_val = metric.get()
    if not isinstance(metric_nm, list):
        metric_nm, metric_val = [metric_nm], [metric_val]

    eval_str = '[Batch %d/%d] loss=%.4f, metrics:' + \
               ','.join([i + ':%.4f' for i in metric_nm])
    logging.info(eval_str, batch_id + 1, batch_num,
                 step_loss / log_interval, *metric_val)


def evaluate(loader_dev, metric, segment):
    """Evaluate the model on validation dataset."""
    logging.info('Now we are doing evaluation on %s with %s.', segment, ctx)
    metric.reset()

    step_loss = 0
    tic = time.time()
    for batch_id, seqs in enumerate(loader_dev):
        input_ids, valid_len, type_ids, label = seqs
        # out = model(
        #     input_ids.as_in_context(ctx), type_ids.as_in_context(ctx),
        #     valid_len.astype('float32').as_in_context(ctx))
        batch = mx.io.DataBatch(data = (input_ids.astype('float32').as_in_context(ctx),
                                        type_ids.astype('float32').as_in_context(ctx),
                                        valid_len.astype('float32').as_in_context(ctx)))
        mod.forward(batch, is_train=False)
        for out in mod.get_outputs():
            out.wait_to_read()
        ls = loss_function(out, label.as_in_context(ctx)).mean()

        step_loss += ls.asscalar()
        metric.update([label], [out])

        if (batch_id + 1) % (log_interval) == 0:
            log_eval(batch_id, len(loader_dev), metric, step_loss, log_interval)
            step_loss = 0

    metric_nm, metric_val = metric.get()
    if not isinstance(metric_nm, list):
        metric_nm, metric_val = [metric_nm], [metric_val]
    metric_str = 'validation metrics:' + ','.join([i + ':%.4f' for i in metric_nm])
    logging.info(metric_str, *metric_val)

    mx.nd.waitall()
    toc = time.time()
    logging.info('Time cost=%.2fs, throughput=%.2f samples/s', toc - tic,
                 dev_batch_size * len(loader_dev) / (toc - tic))


def save_symbol(fname, sym, logger=None):
    if logger is not None:
        logger.info('Saving symbol into file at %s' % fname)
    sym.save(fname)

def save_params(fname, arg_params, aux_params, logger=None):
    if logger is not None:
        logger.info('Saving params into file at %s' % fname)
    save_dict = {('arg:%s' % k): v.as_in_context(cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(cpu()) for k, v in aux_params.items()})
    mx.nd.save(fname, save_dict)

def calibration(loader_dev, segment):
    """Quantize the model on sampling validation dataset."""
    logging.info('Now we are doing calibration on sampling %s with %s.', segment, ctx)

    excluded_sym_names = ['hybridbertmodel0__plus0',
                            'hybridbertencoder0_transformer0__plus0',
                            'hybridbertencoder0_transformer0_bertpositionwiseffn0__plus0',
                            'hybridbertencoder0_transformer1__plus0',
                            'hybridbertencoder0_transformer1_bertpositionwiseffn0__plus0',
                            'hybridbertencoder0_transformer2__plus0',
                            'hybridbertencoder0_transformer2_bertpositionwiseffn0__plus0',
                            'hybridbertencoder0_transformer3__plus0',
                            'hybridbertencoder0_transformer3_bertpositionwiseffn0__plus0',
                            'hybridbertencoder0_transformer4__plus0',
                            'hybridbertencoder0_transformer4_bertpositionwiseffn0__plus0',
                            'hybridbertencoder0_transformer5__plus0',
                            'hybridbertencoder0_transformer5_bertpositionwiseffn0__plus0',
                            'hybridbertencoder0_transformer6__plus0',
                            'hybridbertencoder0_transformer6_bertpositionwiseffn0__plus0',
                            'hybridbertencoder0_transformer7__plus0',
                            'hybridbertencoder0_transformer7_bertpositionwiseffn0__plus0',
                            'hybridbertencoder0_transformer8__plus0',
                            'hybridbertencoder0_transformer8_bertpositionwiseffn0__plus0',
                            'hybridbertencoder0_transformer9__plus0',
                            'hybridbertencoder0_transformer9_bertpositionwiseffn0__plus0',
                            'hybridbertencoder0_transformer10__plus0',
                            'hybridbertencoder0_transformer10_bertpositionwiseffn0__plus0',
                            'hybridbertencoder0_transformer11__plus0',
                            'hybridbertencoder0_transformer11_bertpositionwiseffn0__plus0']

    excluded_sym_names += ['hybridbertmodel0_pooler_tanh_fwd']

    calib_layer = lambda name: name.endswith('_output') or name.endswith('reshape10_0')
    qsym, qarg_params, aux_params = quantize_model(mod=mod, batch_size=dev_batch_size,
                                                   ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                   calib_mode=calib_mode, calib_data=loader_dev,
                                                   num_calib_batch=num_calib_batches,
                                                   calib_layer=calib_layer, quantized_dtype=quantized_dtype,
                                                   logger=logging)
    if calib_mode == 'entropy':
        suffix = '-quantized-entropy'
    elif calib_mode == 'naive':
        suffix = '-quantized-naive'
    elif calib_mode == 'none':
        suffix = '-quantized-none'
    else:
        raise ValueError('unknow calibration mode %s received, only supports `none`, `naive`, and `entropy`'
                         % calib_mode)
    sym_name = '%s-symbol.json' % (model_parameters + suffix)
    if ctx == mx.cpu():
        qsym = qsym.get_backend_symbol('MKLDNN_QUANTIZE')
    save_symbol(sym_name, qsym, logging)
    param_name = '%s-%04d.params' % (model_parameters + suffix, epochs)
    save_params(param_name, qarg_params, aux_params, logging)
    # graph = mx.viz.plot_network(qsym)
    # graph.format = 'png'
    # graph.render(model_parameters + suffix)

if __name__ == '__main__':
    for segment, dev_data in dev_data_list:
        if not only_calibration:
            evaluate(dev_data, task.metrics, segment)
        else:
            calibration(dev_data, segment)

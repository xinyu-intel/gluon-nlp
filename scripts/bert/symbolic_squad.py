"""
SQuAD with Bidirectional Encoder Representations from Transformers

=========================================================================================

This example shows how to implement finetune a model with pre-trained BERT parameters for
SQuAD, with Gluon NLP Toolkit.

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming- \
      Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
"""

# coding=utf-8

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

import argparse
import collections
import json
import logging
import os
import random
import time
import warnings

import numpy as np
import mxnet as mx
from mxnet import gluon, nd

import gluonnlp as nlp
from gluonnlp.data import SQuAD
from bert_qa_model import BertForQALoss, BertForQA
from bert_qa_dataset import (SQuADTransform, preprocess_dataset)
from bert_qa_evaluate import get_F1_EM, predictions
from quantization import *

np.random.seed(6)
random.seed(6)
mx.random.seed(6)

log = logging.getLogger('gluonnlp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s', datefmt='%H:%M:%S')


parser = argparse.ArgumentParser(description='BERT QA example.'
                                 'We fine-tune the BERT model on SQuAD dataset.')

parser.add_argument('--only-calibration',
                    action='store_true',
                    help='Whether to predict only.')

parser.add_argument('--model_parameters',
                    type=str,
                    default=None,
                    help='Model parameter file')

parser.add_argument('--bert_model',
                    type=str,
                    default='bert_12_768_12',
                    help='BERT model name. options are bert_12_768_12 and bert_24_1024_16.')

parser.add_argument('--bert_dataset',
                    type=str,
                    default='book_corpus_wiki_en_uncased',
                    help='BERT dataset name.'
                    'options are book_corpus_wiki_en_uncased and book_corpus_wiki_en_cased.')

parser.add_argument('--pretrained_bert_parameters',
                    type=str,
                    default=None,
                    help='Pre-trained bert model parameter file. default is None')

parser.add_argument('--uncased',
                    action='store_false',
                    help='if not set, inputs are converted to lower case.')

parser.add_argument('--output_dir',
                    type=str,
                    default='./output_dir',
                    help='The output directory where the model params will be written.'
                    ' default is ./output_dir')

parser.add_argument('--epochs',
                    type=int,
                    default=0,
                    help='number of epochs, default is 3')

parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help='Batch size. Number of examples per gpu in a minibatch. default is 32')

parser.add_argument('--test_batch_size',
                    type=int,
                    default=24,
                    help='Test batch size. default is 24')

parser.add_argument('--optimizer',
                    type=str,
                    default='bertadam',
                    help='optimization algorithm. default is bertadam(mxnet >= 1.5.0.)')

parser.add_argument('--accumulate',
                    type=int,
                    default=None,
                    help='The number of batches for '
                    'gradients accumulation to simulate large batch size. Default is None')

parser.add_argument('--lr',
                    type=float,
                    default=5e-5,
                    help='Initial learning rate. default is 5e-5')

parser.add_argument('--warmup_ratio',
                    type=float,
                    default=0.1,
                    help='ratio of warmup steps that linearly increase learning rate from '
                    '0 to target learning rate. default is 0.1')

parser.add_argument('--log_interval',
                    type=int,
                    default=50,
                    help='report interval. default is 50')

parser.add_argument('--max_seq_length',
                    type=int,
                    default=384,
                    help='The maximum total input sequence length after WordPiece tokenization.'
                    'Sequences longer than this will be truncated, and sequences shorter '
                    'than this will be padded. default is 384')

parser.add_argument('--doc_stride',
                    type=int,
                    default=128,
                    help='When splitting up a long document into chunks, how much stride to '
                    'take between chunks. default is 128')

parser.add_argument('--max_query_length',
                    type=int,
                    default=64,
                    help='The maximum number of tokens for the question. Questions longer than '
                    'this will be truncated to this length. default is 64')

parser.add_argument('--n_best_size',
                    type=int,
                    default=20,
                    help='The total number of n-best predictions to generate in the '
                    'nbest_predictions.json output file. default is 20')

parser.add_argument('--max_answer_length',
                    type=int,
                    default=30,
                    help='The maximum length of an answer that can be generated. This is needed '
                    'because the start and end predictions are not conditioned on one another.'
                    ' default is 30')

parser.add_argument('--version_2',
                    action='store_true',
                    help='SQuAD examples whether contain some that do not have an answer.')

parser.add_argument('--null_score_diff_threshold',
                    type=float,
                    default=0.0,
                    help='If null_score - best_non_null is greater than the threshold predict null.'
                    'Typical values are between -1.0 and -5.0. default is 0.0')

parser.add_argument('--gpu',
                    action='store_true',
                    help='whether to use gpu for finetuning')

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

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

fh = logging.FileHandler(os.path.join(
    args.output_dir, 'finetune_squad.log'), mode='w')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
log.addHandler(console)
log.addHandler(fh)

log.info(args)

model_name = args.bert_model
dataset_name = args.bert_dataset
only_calibration = args.only_calibration
model_parameters = args.model_parameters
pretrained_bert_parameters = args.pretrained_bert_parameters
lower = args.uncased

epochs = args.epochs
batch_size = args.batch_size
test_batch_size = args.test_batch_size
lr = args.lr
ctx = mx.cpu() if not args.gpu else mx.gpu()

accumulate = args.accumulate
log_interval = args.log_interval * accumulate if accumulate else args.log_interval
if accumulate:
    log.info('Using gradient accumulation. Effective batch size = {}'.
             format(accumulate*batch_size))

optimizer = args.optimizer
warmup_ratio = args.warmup_ratio


version_2 = args.version_2
null_score_diff_threshold = args.null_score_diff_threshold

max_seq_length = args.max_seq_length
doc_stride = args.doc_stride
max_query_length = args.max_query_length
n_best_size = args.n_best_size
max_answer_length = args.max_answer_length

# calibration
num_calib_batches = args.num_calib_batches
calib_mode = args.calib_mode
quantized_dtype = args.quantized_dtype

if max_seq_length <= max_query_length + 3:
    raise ValueError('The max_seq_length (%d) must be greater than max_query_length '
                     '(%d) + 3' % (max_seq_length, max_query_length))

bert, vocab = nlp.model.get_model(
    name=model_name,
    dataset_name=dataset_name,
    pretrained=not model_parameters and not pretrained_bert_parameters,
    ctx=ctx,
    use_pooler=False,
    use_decoder=False,
    use_classifier=False)

batchify_fn = nlp.data.batchify.Tuple(
    nlp.data.batchify.Stack(),
    nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token]),
    nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token]),
    nlp.data.batchify.Stack('float32'),
    nlp.data.batchify.Stack('float32'),
    nlp.data.batchify.Stack('float32'))

berttoken = nlp.data.BERTTokenizer(vocab=vocab, lower=lower)

# net = BertForQA(bert=bert)
# if pretrained_bert_parameters and not model_parameters:
#     bert.load_parameters(pretrained_bert_parameters, ctx=ctx,
#                          ignore_extra=True)
# if not model_parameters:
#     net.span_classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
# else:
#     net.load_parameters(model_parameters, ctx=ctx)
# net.hybridize(static_alloc=True)

# loss_function = BertForQALoss()
# loss_function.hybridize(static_alloc=True)

model_prefix = model_parameters
network, args_, auxs = mx.model.load_checkpoint(model_prefix, epochs)
if ctx == mx.cpu():
    network = network.get_backend_symbol('MKLDNN')
data_shape0 = (test_batch_size, max_seq_length)
data_shape1 = (test_batch_size, max_seq_length)
data_shape2 = (test_batch_size,)
data_names = ('data0', 'data1', 'data2')
mod = mx.mod.Module(symbol=network,  context=ctx, data_names=data_names, label_names=None)
mod.bind(for_training=False,
            data_shapes=[('data0', data_shape0), ('data1', data_shape1), ('data2', data_shape2)])
mod.set_params(args_, auxs, allow_missing=False, force_init=True)

def evaluate():
    """Evaluate the model on validation dataset.
    """
    log.info('Loader dev data...')
    if version_2:
        dev_data = SQuAD('dev', version='2.0')
    else:
        dev_data = SQuAD('dev', version='1.1')
    log.info('Number of records in Train data:{}'.format(len(dev_data)))

    dev_dataset = dev_data.transform(
        SQuADTransform(
            berttoken,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_pad=True,
            is_training=False)._transform)

    dev_data_transform, _ = preprocess_dataset(
        dev_data, SQuADTransform(
            berttoken,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_pad=True,
            is_training=False))
    log.info('The number of examples after preprocessing:{}'.format(
        len(dev_data_transform)))

    dev_dataloader = mx.gluon.data.DataLoader(
        dev_data_transform,
        batchify_fn=batchify_fn,
        num_workers=4, batch_size=test_batch_size, shuffle=False, last_batch='keep')

    log.info('Start predict')

    _Result = collections.namedtuple(
        '_Result', ['example_id', 'start_logits', 'end_logits'])
    all_results = {}

    epoch_tic = time.time()
    total_num = 0
    for data in dev_dataloader:
        example_ids, inputs, token_types, valid_length, _, _ = data
        total_num += len(inputs)
        # out = net(inputs.astype('float32').as_in_context(ctx),
        #           token_types.astype('float32').as_in_context(ctx),
        #           valid_length.astype('float32').as_in_context(ctx))

        batch = mx.io.DataBatch(data = (inputs.astype('float32').as_in_context(ctx),
                                        token_types.astype('float32').as_in_context(ctx),
                                        valid_length.astype('float32').as_in_context(ctx)))
        mod.forward(batch, is_train=False)
        for out in mod.get_outputs():
            out.wait_to_read()
        output = nd.split(out, axis=2, num_outputs=2)
        start_logits = output[0].reshape((0, -3)).asnumpy()
        end_logits = output[1].reshape((0, -3)).asnumpy()

        for example_id, start, end in zip(example_ids, start_logits, end_logits):
            example_id = example_id.asscalar()
            if example_id not in all_results:
                all_results[example_id] = []
            all_results[example_id].append(
                _Result(example_id, start.tolist(), end.tolist()))
    epoch_toc = time.time()
    log.info('Time cost={:.2f} s, Thoughput={:.2f} samples/s'.format(
        epoch_toc - epoch_tic, total_num/(epoch_toc - epoch_tic)))
    log.info('Get prediction results...')

    all_predictions, all_nbest_json, scores_diff_json = predictions(
        dev_dataset=dev_dataset,
        all_results=all_results,
        tokenizer=nlp.data.BERTBasicTokenizer(lower=lower),
        max_answer_length=max_answer_length,
        null_score_diff_threshold=null_score_diff_threshold,
        n_best_size=n_best_size,
        version_2=version_2)

    with open(os.path.join(output_dir, 'predictions.json'),
              'w', encoding='utf-8') as all_predictions_write:
        all_predictions_write.write(json.dumps(all_predictions))

    with open(os.path.join(output_dir, 'nbest_predictions.json'),
              'w', encoding='utf-8') as all_predictions_write:
        all_predictions_write.write(json.dumps(all_nbest_json))

    if version_2:
        with open(os.path.join(output_dir, 'null_odds.json'),
                  'w', encoding='utf-8') as all_predictions_write:
            all_predictions_write.write(json.dumps(scores_diff_json))
    else:
        log.info(get_F1_EM(dev_data, all_predictions))

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

def calibration():
    """Calibration the model on validation dataset.
    """
    log.info('Loader dev data...')
    if version_2:
        dev_data = SQuAD('dev', version='2.0')
    else:
        dev_data = SQuAD('dev', version='1.1')
    log.info('Number of records in Train data:{}'.format(len(dev_data)))

    dev_dataset = dev_data.transform(
        SQuADTransform(
            berttoken,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_pad=True,
            is_training=False)._transform)

    dev_data_transform, _ = preprocess_dataset(
        dev_data, SQuADTransform(
            berttoken,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_pad=True,
            is_training=False))
    log.info('The number of examples after preprocessing:{}'.format(
        len(dev_data_transform)))

    dev_dataloader = mx.gluon.data.DataLoader(
        dev_data_transform,
        batchify_fn=batchify_fn,
        num_workers=4, batch_size=test_batch_size, shuffle=False, last_batch='keep')

    log.info('Start Calibration')

    excluded_sym_names = ['hybridbertencoder0_transformer0_bertpositionwiseffn0_ffn_2_fwd',
                          'hybridbertencoder0_transformer1_bertpositionwiseffn0_ffn_2_fwd',
                          'hybridbertencoder0_transformer2_bertpositionwiseffn0_ffn_2_fwd',
                          'hybridbertencoder0_transformer3_bertpositionwiseffn0_ffn_2_fwd',
                          'hybridbertencoder0_transformer4_bertpositionwiseffn0_ffn_2_fwd',
                          'hybridbertencoder0_transformer5_bertpositionwiseffn0_ffn_2_fwd',
                          'hybridbertencoder0_transformer6_bertpositionwiseffn0_ffn_2_fwd',
                          'hybridbertencoder0_transformer7_bertpositionwiseffn0_ffn_2_fwd',
                          'hybridbertencoder0_transformer8_bertpositionwiseffn0_ffn_2_fwd',
                          'hybridbertencoder0_transformer9_bertpositionwiseffn0_ffn_2_fwd',
                          'hybridbertencoder0_transformer10_bertpositionwiseffn0_ffn_2_fwd',
                          'hybridbertencoder0_transformer11_bertpositionwiseffn0_ffn_2_fwd',
                          'staticbertforqa0_dense0_fwd']

    excluded_sym_names += ['hybridbertmodel0__plus0',
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

    calib_layer = lambda name: name.endswith('_output') or name.endswith('reshape10_0')
    qsym, qarg_params, aux_params = quantize_model(mod=mod, batch_size=test_batch_size,
                                                   ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                   calib_mode=calib_mode, calib_data=dev_dataloader,
                                                   num_calib_batch=num_calib_batches,
                                                   calib_layer=calib_layer, 
                                                   model='squad', quantized_dtype=quantized_dtype,
                                                   logger=log)
    if calib_mode == 'entropy':
        suffix = '-quantized-entropy'
    elif calib_mode == 'naive':
        suffix = '-quantized-naive'
    elif calib_mode == 'none':
        suffix = '-quantized-none'
    else:
        raise ValueError('unknow calibration mode %s received, only supports `none`, `naive`, and `entropy`'
                         % calib_mode)
    sym_name = '%s-symbol.json' % (model_prefix + suffix)
    if ctx == mx.cpu():
        qsym = qsym.get_backend_symbol('MKLDNN_QUANTIZE')
    save_symbol(sym_name, qsym, log)
    param_name = '%s-%04d.params' % (model_prefix + suffix, epochs)
    save_params(param_name, qarg_params, aux_params, log)
    # graph = mx.viz.plot_network(qsym)
    # graph.format = 'png'
    # graph.render(model_prefix + suffix)

if __name__ == '__main__':
    if only_calibration:
        calibration()
    else:
        evaluate()

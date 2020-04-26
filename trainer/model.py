#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import WALSMatrixFactorization

tf.logging.set_verbosity(tf.logging.INFO)

import os
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.contrib.factorization import WALSMatrixFactorization
  
import os
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.contrib.factorization import WALSMatrixFactorization
  
def read_dataset(mode, args):
    def decode_example(protos, vocab_size):
        features = {
            "key": tf.FixedLenFeature(shape = [1], dtype = tf.int64),
            "indices": tf.VarLenFeature(dtype = tf.int64),
            "values": tf.VarLenFeature(dtype = tf.float32)}
        parsed_features = tf.parse_single_example(serialized = protos, features = features)
        values = tf.sparse_merge(sp_ids = parsed_features["indices"], sp_values = parsed_features["values"], vocab_size = vocab_size)
        # Save key to remap after batching
        # This is a temporary workaround to assign correct row numbers in each batch.
        # You can ignore details of this part and remap_keys().
        key = parsed_features["key"]
        decoded_sparse_tensor = tf.SparseTensor(indices = tf.concat(values = [values.indices, [key]], axis = 0), 
                                                values = tf.concat(values = [values.values, [0.0]], axis = 0), 
                                                dense_shape = values.dense_shape)
        return decoded_sparse_tensor
  
  
    def remap(sparse_tensor):
        bad_indices = sparse_tensor.indices 
        bad_values = sparse_tensor.values 
        size = tf.segment_sum(data = tf.ones_like(bad_indices[:,0], dtype = tf.int64),
                           segment_ids = bad_indices[:, 0]) - 1
        length = tf.shape(size, out_type = tf.int64)[0]
        cum = tf.cumsum(size)
        #offset between each example in the batch due to concatenation of the key in decode fn
        length_range = tf.range(start = 0, limit = length, delta = 1, dtype = tf.int64)
        cum_range = cum + length_range
        gathered_indices = tf.squeeze(tf.gather(bad_indices, cum_range)[:, 1])
        sparse_indices_range = tf.range(tf.shape(bad_indices, out_type = tf.int64)[0], dtype = tf.int64)
        # find the rows that are not keys
        #
        x = sparse_indices_range
        # indices of keys
        s = cum_range
        tile_multiples = tf.concat([tf.ones(tf.shape(tf.shape(x)), dtype = tf.int64),
                                   tf.shape(s, out_type = tf.int64)], axis = 0)
        x_tile = tf.tile(tf.expand_dims(x, -1), tile_multiples)
        x_not_in_s = ~tf.reduce_any(tf.equal(x_tile, s), -1)
        selected_indices = tf.boolean_mask(tensor = bad_indices, mask = x_not_in_s, axis = 0)
        selected_values = tf.boolean_mask(tensor = bad_values, mask = x_not_in_s, axis = 0)
        tiling = tf.tile(input = tf.expand_dims(gathered_indices[0], -1),
                        multiples = tf.expand_dims(size[0], -1))
        def loop_body(i, tensor_grow):
            return i+1, tf.concat(values = [tensor_grow,
                                            tf.tile(input = tf.expand_dims(gathered_indices[i], -1),
                                                   multiples = tf.expand_dims(size[i], -1))         
                                           ], axis = 0)
        _, result = tf.while_loop(lambda i, tensor_grow: i < length, loop_body, 
                                  [tf.constant(1, dtype = tf.int64), tiling])
        selected_indices_fixed = tf.concat([tf.expand_dims(result, -1),
                                            tf.expand_dims(selected_indices[:, 1], -1)],
                                            axis = 1)
        remapped_sparse_tensor = tf.SparseTensor(indices = selected_indices_fixed, 
                                                values = selected_values,
                                                dense_shape = sparse_tensor.dense_shape)
        return remapped_sparse_tensor

    
    def parse_tfrecords(filename, vocab_size):
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
        else:
            num_epochs = 1 # end-of-input after this

        files = tf.gfile.Glob(filename = os.path.join(args["input_path"], filename))

        # Create dataset from file list
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(map_func = lambda x: decode_example(x, vocab_size))
        dataset = dataset.repeat(count = num_epochs)
        dataset = dataset.batch(batch_size = args["batch_size"])
        dataset = dataset.map(map_func = lambda x: remap(x))
        return dataset.make_one_shot_iterator().get_next()
  
    def _input_fn():
        features = {
            WALSMatrixFactorization.INPUT_ROWS: parse_tfrecords("items_for_user", args["nitems"]),
            WALSMatrixFactorization.INPUT_COLS: parse_tfrecords("users_for_item", args["nusers"]),
            WALSMatrixFactorization.PROJECT_ROW: tf.constant(True)
        }
        return features, None

    return _input_fn
  
    def input_cols():
        return parse_tfrecords('users_for_item', args['nusers'])
  
    return _input_fn

def find_top_k(user, item_factors, k):
    all_items = tf.matmul(a = tf.expand_dims(input = user, axis = 0), b = tf.transpose(a = item_factors))
    topk = tf.nn.top_k(input = all_items, k = k)
    return tf.cast(x = topk.indices, dtype = tf.int64)
    
def batch_predict(args):
    def create_lookup(filename):
        from tensorflow.python.lib.io import file_io
        dirname = os.path.join(args["input_path"], filename)
        with file_io.FileIO(dirname, mode = 'r') as ifp:
            return [x.rstrip() for x in ifp]
    itemIds = create_lookup("items.csv")
    userIds = create_lookup("users.csv")
    import numpy as np
    with tf.Session() as sess:
        estimator = tf.contrib.factorization.WALSMatrixFactorization(
            num_rows = args["nusers"], 
            num_cols = args["nitems"],
            embedding_dimension = args["n_embeds"],
            model_dir = args["output_dir"])
        
        # This is how you would get the row factors for out-of-vocab user data
        # row_factors = list(estimator.get_projections(input_fn=read_dataset(tf.estimator.ModeKeys.EVAL, args)))
        # user_factors = tf.convert_to_tensor(np.array(row_factors))

        # But for in-vocab data, the row factors are already in the checkpoint
        user_factors = tf.convert_to_tensor(value = estimator.get_row_factors()[0]) # (nusers, nembeds)
        # In either case, we have to assume catalog doesn"t change, so col_factors are read in
        item_factors = tf.convert_to_tensor(value = estimator.get_col_factors()[0])# (nitems, nembeds)

        # For each user, find the top K items
        topk = tf.squeeze(input = tf.map_fn(fn = lambda user: find_top_k(user, item_factors, args["topk"]), elems = user_factors, dtype = tf.int64))
        with file_io.FileIO(os.path.join(args["output_dir"], "batch_pred.csv"), mode = 'w') as f:
            for userId, best_items_for_user in enumerate(topk.eval()):
                f.write(userIds[userId].split(",")[0] + ",")
                f.write(",".join(itemIds[x].split(",")[0] for x in best_items_for_user) + '\n')

def train_and_evaluate(args):
    train_steps = int(0.5 + (1.0 * args["num_epochs"] * args["nusers"]) / args["batch_size"])
    steps_in_epoch = int(0.5 + args["nusers"] / args["batch_size"])
    print("Will train for {} steps, evaluating once every {} steps".format(train_steps, steps_in_epoch))
    def experiment_fn(output_dir):
        return tf.contrib.learn.Experiment(
            tf.contrib.factorization.WALSMatrixFactorization(
                num_rows = args["nusers"], 
                num_cols = args["nitems"],
                embedding_dimension = args["n_embeds"],
                model_dir = args["output_dir"]),
            train_input_fn = read_dataset(tf.estimator.ModeKeys.TRAIN, args),
            eval_input_fn = read_dataset(tf.estimator.ModeKeys.EVAL, args),
            train_steps = train_steps,
            eval_steps = 1,
            min_eval_frequency = steps_in_epoch
        )

    from tensorflow.contrib.learn.python.learn import learn_runner
    learn_runner.run(experiment_fn = experiment_fn, output_dir = args["output_dir"])
    
    batch_predict(args)
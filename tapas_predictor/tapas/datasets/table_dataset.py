# coding=utf-8
# Copyright 2019 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""Contains parse functions for table inputs."""

from tapas.datasets import dataset
from tapas.utils import text_utils
import tensorflow.compat.v1 as tf


def parse_table_examples(max_seq_length, max_predictions_per_seq,
                         is_pretraining,
                         add_aggregation_function_id,
                         add_classification_labels, add_answer,
                         include_id, add_candidate_answers,
                         max_num_candidates,
                         params):
  """Returns a parse_fn that parses tf.Example in table format."""

  print("Calling me over and again :")
  print("########### parse_table_examples - Params ###########")
  print("\n max_seq_length                  : ",max_seq_length, 
        "\n max_predictions_per_seq         : ",max_predictions_per_seq,
        "\n is_pretraining                  : ",is_pretraining,
        "\n add_aggregation_function_id     : ",add_aggregation_function_id,
        "\n add_classification_labels       : ",add_classification_labels, add_answer,
        "\n include_id                      : ",include_id, 
        "\n add_candidate_answers           : ",add_candidate_answers,
        "\n max_num_candidates              : ",max_num_candidates,
        "\n params                          : ",params)
  print("\n########### parse_table_examples - Params ###########")

  feature_types = {
      "input_ids":
          tf.FixedLenFeature([max_seq_length], tf.int64),
      "input_mask":
          tf.FixedLenFeature([max_seq_length], tf.int64),
      "segment_ids":
          tf.FixedLenFeature([max_seq_length], tf.int64),
      "column_ids":
          tf.FixedLenFeature([max_seq_length], tf.int64),
      "row_ids":
          tf.FixedLenFeature([max_seq_length], tf.int64),
      "prev_label_ids":
          tf.FixedLenFeature([max_seq_length], tf.int64),
      "column_ranks":
          tf.FixedLenFeature([max_seq_length], tf.int64),
      "inv_column_ranks":
          tf.FixedLenFeature([max_seq_length], tf.int64),
      "numeric_relations":
          tf.FixedLenFeature([max_seq_length],
                             tf.int64,
                             default_value=[0] * max_seq_length),
  }

  if is_pretraining:
    feature_types.update({
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    })
  else:
    # For classification we have a label for each token.
    feature_types.update({
        "label_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
    })
    # TODO Remove default_value once the data has been updated.
    feature_types.update({
        "question_id_ints":
            tf.FixedLenFeature([text_utils.DEFAULT_INTS_LENGTH],
                               tf.int64,
                               default_value=[0] *
                               text_utils.DEFAULT_INTS_LENGTH),
    })
    # Label for predicting the aggregation function.
    if add_aggregation_function_id:
      feature_types.update({
          "aggregation_function_id": tf.FixedLenFeature([1], tf.int64),
      })
    if add_classification_labels:
      feature_types.update({
          "classification_class_index": tf.FixedLenFeature([1], tf.int64),
      })
    # Features for the weakly supervised setting.
    if add_answer:
      feature_types.update({
          "numeric_values":
              tf.FixedLenFeature([max_seq_length], tf.float32),
          "numeric_values_scale":
              tf.FixedLenFeature([max_seq_length], tf.float32),
          "answer":
              tf.FixedLenFeature([1], tf.float32),
      })

  if add_candidate_answers:
    feature_types.update({
        "cand_num":
            tf.FixedLenFeature([], tf.int64),
        "can_aggregation_function_ids":
            tf.FixedLenFeature([max_num_candidates], tf.int64),
        "can_sizes":
            tf.FixedLenFeature([max_num_candidates], tf.int64),
        "can_indexes":
            tf.VarLenFeature(tf.int64),
    })

  if include_id:
    feature_types.update({
        "question_id": tf.FixedLenFeature([1], tf.string),
    })

  def _parse_fn(serialized_example):
    print("Calling serialized_example table_dataset.py**", serialized_example, add_candidate_answers)
    
    features = dict(
        dataset.build_parser_function(feature_types,
                                      params)(serialized_example))
    print("****Features Built from dataset*************")
    print(features)
    print("******************")
    if add_candidate_answers:
      _preprocess_candidate_answers(
          features,
          max_num_candidates=max_num_candidates,
          max_seq_length=max_seq_length)
    return features

  return _parse_fn


def _preprocess_candidate_answers(features, max_num_candidates, max_seq_length):
  """Prepares dense labels for each candidate."""
  ragged_indices = tf.RaggedTensor.from_row_lengths(
      features["can_indexes"].values, features["can_sizes"])
  candidate_id = tf.ragged.row_splits_to_segment_ids(ragged_indices.row_splits)
  indices = tf.stack([candidate_id, ragged_indices.flat_values], axis=-1)
  updates = tf.ones_like(candidate_id, dtype=tf.int32)
  features["can_label_ids"] = tf.scatter_nd(
      indices=indices,
      updates=updates,
      shape=[max_num_candidates, max_seq_length])
  # Variable length tensors are not supported on TPU.
  del features["can_indexes"]

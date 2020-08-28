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
"""Writes prediction to a csv file."""

import collections
import copy
import csv
from typing import Mapping, Text, Tuple, Iterable

from absl import logging
import numpy as np
from tapas.models import tapas_classifier_model
from tapas.utils import text_utils
import tensorflow.compat.v1 as tf


#Moidified for avoiding TF Dataset
from tapas.utils import tf_example_utils
from tapas.protos import interaction_pb2
from tapas.utils import number_annotation_utils

def predict_modified_json(table_data, queries):

  max_seq_length = 512
  vocab_file = "tapas_sqa_base/vocab.txt"
  config = tf_example_utils.ClassifierConversionConfig(
      vocab_file=vocab_file,
      max_seq_length=max_seq_length,
      max_column_id=max_seq_length,
      max_row_id=max_seq_length,
      strip_column_names=False,
      add_aggregation_candidates=False,
  )
  converter = tf_example_utils.ToClassifierTensorflowExample(config)

  def convert_interactions_to_examples(tables_and_queries):
    """Calls Tapas converter to convert interaction to example."""
    print("inside")
    for idx, (table, queries) in enumerate(tables_and_queries):
      print("inside for")
      interaction = interaction_pb2.Interaction()
      print("interaction : ", interaction)
      
      for position, query in enumerate(queries):
        print(position, queries)
        question = interaction.questions.add()
        question.original_text = query
        question.id = f"{idx}-0_{position}"
      for header in table[0]:
        print("Header : ",header)
        interaction.table.columns.add().text = header
      for line in table[1:]:
        row = interaction.table.rows.add()
        print("row", line)
        for cell in line:
          print("cell : ")
          row.cells.add().text = cell
      print("interaction", interaction)
      number_annotation_utils.add_numeric_values(interaction)
      for i in range(len(interaction.questions)):
        print(i)
        try:
          yield converter.convert(interaction, i)
        except ValueError as e:
          print(f"Can't convert interaction: {interaction.id} error: {e}")



  table = [list(map(lambda s: s.strip(), row.split("|"))) 
           for row in table_data.split("\n") if row.strip()]

  examples = convert_interactions_to_examples([(table, queries)])
  return examples



def predict_modified(table_data, queries):

  max_seq_length = 512
  vocab_file = "tapas_sqa_base/vocab.txt"
  config = tf_example_utils.ClassifierConversionConfig(
      vocab_file=vocab_file,
      max_seq_length=max_seq_length,
      max_column_id=max_seq_length,
      max_row_id=max_seq_length,
      strip_column_names=False,
      add_aggregation_candidates=False,
  )
  converter = tf_example_utils_eldhose.ToClassifierTensorflowExample(config)

  def convert_interactions_to_examples(tables_and_queries):
    """Calls Tapas converter to convert interaction to example."""
    print("inside")
    for idx, (table, queries) in enumerate(tables_and_queries):
      print("inside for")
      interaction = interaction_pb2.Interaction()
      print("interaction : ", interaction)
      
      for position, query in enumerate(queries):
        print(position, queries)
        question = interaction.questions.add()
        question.original_text = query
        question.id = f"{idx}-0_{position}"
      for header in table[0]:
        print("Header : ",header)
        interaction.table.columns.add().text = header
      for line in table[1:]:
        row = interaction.table.rows.add()
        print("row", line)
        for cell in line:
          print("cell : ")
          row.cells.add().text = cell
      print("interaction", interaction)
      number_annotation_utils.add_numeric_values(interaction)
      for i in range(len(interaction.questions)):
        print(i)
        try:
          yield converter.convert(interaction, i)
        except ValueError as e:
          print(f"Can't convert interaction: {interaction.id} error: {e}")



  table = [list(map(lambda s: s.strip(), row.split("|"))) 
           for row in table_data.split("\n") if row.strip()]
  examples = convert_interactions_to_examples([(table, queries)])
  return examples




def _get_in_memory_input_fn(examples):
  """An input function that reads examples from numpy arrays in memory."""

  def _input_fn(params):
    """The input function."""
    print("Examples from Input Function : ")
    print("Examples Length : ",len(examples))
    print("Example ID: ")
    features = collections.defaultdict(list)
    for example in examples:
      data_tf = tf.convert_to_tensor(example['label_ids'], np.int32)
      print("data_tf  -  ",type(data_tf),data_tf.shape)
      for feature_key in example:
        try:
          print("feature_key : ", feature_key, type(example[feature_key]),example[feature_key].shape)
          features[feature_key].append(example[feature_key])
          # testing_a = np.concatenate(features[feature_key], axis=0)
        except Exception as e:
          print("feature_key : ", feature_key, type(example[feature_key]))
          # print("testing_a : ", testing_a.shape, testing_a)
    dataset = tf.data.Dataset.zip({
        feature_key: tf.data.Dataset.from_tensor_slices(
            np.concatenate(features[feature_key], axis=0))
        for feature_key in features
    })
    # exit(0)
    return dataset.batch(params["batch_size"])

  return _input_fn


def compute_prediction_sequence(estimator, examples_by_position):
  """Computes predictions using model's answers to the previous questions."""
  all_results = []
  prev_answers = None

  for position in range(len(examples_by_position)):
    print("Positions Line 113 : ", position)
    examples = copy.deepcopy(examples_by_position[position])
    if prev_answers is not None:
      for example_id in examples:
        print("Example ID Previous Answer is not None : ", example_id)
        coords_to_answer = prev_answers[example_id]
        example = examples[example_id]
        prev_label_ids = example["prev_label_ids"]
        model_label_ids = np.zeros_like(prev_label_ids)
        for i in range(model_label_ids.shape[1]):
          row_id = example["row_ids"][0, i] - 1
          col_id = example["column_ids"][0, i] - 1
          if row_id >= 0 and col_id >= 0 and example["segment_ids"][0, i] == 1:
            model_label_ids[0, i] = int(coords_to_answer[(col_id, row_id)])
        examples[example_id]["prev_label_ids"] = model_label_ids
    print(type(estimator))
    # exit(0)
    print("CALLING ESTIMATOR RECURSIVELY TO GENERATE A LIST")
    results = list(
        estimator.predict(input_fn=_get_in_memory_input_fn(examples.values())))

    print("Verifying Results :", results)
    all_results.extend(results)

    print("Size of all Lists : ", len(results))
    prev_answers = {}
    for prediction in results:
      question_id = prediction["question_id"][0].decode("utf-8")
      table_id, annotator, _ = text_utils.parse_question_id(question_id)
      example_id = (table_id, annotator)
      example = examples[example_id]
      probabilities = prediction["probabilities"]

      # Compute average probability per cell, aggregating over tokens.
      coords_to_probs = collections.defaultdict(list)
      for i, p in enumerate(probabilities):
        segment_id = prediction["segment_ids"][i]
        col = prediction["column_ids"][i] - 1
        row = prediction["row_ids"][i] - 1
        if col >= 0 and row >= 0 and segment_id == 1:
          coords_to_probs[(col, row)].append(p)

      coords_to_answer = {}
      for key in coords_to_probs:
        coords_to_answer[key] = np.array(coords_to_probs[key]).mean() > 0.5
      prev_answers[example_id] = coords_to_answer

  return all_results


def _get_question_id(features):
  """Restores question id from int sequence."""
  if "question_id_ints" in features:
    question_id = text_utils.ints_to_str(features["question_id_ints"])
    if question_id:
      return question_id
  # TODO Remove once the data has been updated.
  return features["question_id"][0].decode("utf-8")


def get_mean_cell_probs(prediction):
  """Computes average probability per cell, aggregating over tokens."""
  probabilities = prediction["probabilities"]
  coords_to_probs = collections.defaultdict(list)
  for i, p in enumerate(probabilities):
    segment_id = prediction["segment_ids"][i]
    col = prediction["column_ids"][i] - 1
    row = prediction["row_ids"][i] - 1
    if col >= 0 and row >= 0 and segment_id == 1:
      coords_to_probs[(col, row)].append(p)
  return {
      coords: np.array(cell_probs).mean()
      for coords, cell_probs in coords_to_probs.items()
  }


def write_predictions(
    predictions,
    output_predict_file,
    do_model_aggregation,
    do_model_classification,
    cell_classification_threshold,
):
  """Writes predictions to an output TSV file.

  Predictions header: [id, annotator, position, answer_coordinates, gold_aggr,
  pred_aggr]

  Args:
    predictions: model predictions
    output_predict_file: Path for wrinting the predicitons.
    do_model_aggregation: Indicates whther to write predicted aggregations.
    do_model_classification: Indicates whther to write predicted classes.
    cell_classification_threshold: Threshold for selecting a cell.
  """
  with tf.io.gfile.GFile(output_predict_file, "w") as write_file:
    header = [
        "question_id",
        "id",
        "annotator",
        "position",
        "answer_coordinates",
    ]
    if do_model_aggregation:
      header.extend(["gold_aggr", "pred_aggr"])
    if do_model_classification:
      header.extend(["gold_cls", "pred_cls", "logits_cls"])
    writer = csv.DictWriter(write_file, fieldnames=header, delimiter="\t")
    writer.writeheader()

    for prediction in predictions:
      question_id = _get_question_id(prediction)
      max_width = prediction["column_ids"].max()
      max_height = prediction["row_ids"].max()

      if (max_width == 0 and max_height == 0 and
          question_id == text_utils.get_padded_question_id()):
        logging.info("Removing padded example: %s", question_id)
        continue

      cell_coords_to_prob = get_mean_cell_probs(prediction)

      # Select the answers above a classification threshold.
      answer_coordinates = []
      for col in range(max_width):
        for row in range(max_height):
          cell_prob = cell_coords_to_prob.get((col, row), None)
          if cell_prob is not None:
            if cell_prob > cell_classification_threshold:
              answer_coordinates.append(str((row, col)))

      try:
        example_id, annotator, position = text_utils.parse_question_id(
            question_id)
        position = str(position)
      except ValueError:
        example_id = "_"
        annotator = "_"
        position = "_"
      prediction_to_write = {
          "question_id": question_id,
          "id": example_id,
          "annotator": annotator,
          "position": position,
          "answer_coordinates": str(answer_coordinates)
      }
      if do_model_aggregation:
        prediction_to_write["gold_aggr"] = str(prediction["gold_aggr"][0])
        prediction_to_write["pred_aggr"] = str(prediction["pred_aggr"])
      if do_model_classification:
        prediction_to_write["gold_cls"] = str(prediction["gold_cls"][0])
        prediction_to_write["pred_cls"] = str(prediction["pred_cls"])
        prediction_to_write["logits_cls"] = str(prediction["logits_cls"])
      writer.writerow(prediction_to_write)


def retrieve_predictions(
    predictions,
    do_model_aggregation,
    do_model_classification,
    cell_classification_threshold,
):
  """Writes predictions to an output TSV file.

  Predictions header: [id, annotator, position, answer_coordinates, gold_aggr,
  pred_aggr]

  Args:
    predictions: model predictions
    output_predict_file: Path for wrinting the predicitons.
    do_model_aggregation: Indicates whther to write predicted aggregations.
    do_model_classification: Indicates whther to write predicted classes.
    cell_classification_threshold: Threshold for selecting a cell.
  """
  # with tf.io.gfile.GFile(output_predict_file, "w") as write_file:
  result_predicted = []
  header = [
      "question_id",
      "id",
      "annotator",
      "position",
      "answer_coordinates",
  ]
  if do_model_aggregation:
    header.extend(["gold_aggr", "pred_aggr"])
  if do_model_classification:
    header.extend(["gold_cls", "pred_cls", "logits_cls"])
  # writer = csv.DictWriter(write_file, fieldnames=header, delimiter="\t")
  # writer.writeheader()
  result_predicted.append(header)

  for prediction in predictions:
    question_id = _get_question_id(prediction)
    max_width = prediction["column_ids"].max()
    max_height = prediction["row_ids"].max()

    if (max_width == 0 and max_height == 0 and
        question_id == text_utils.get_padded_question_id()):
      logging.info("Removing padded example: %s", question_id)
      continue

    cell_coords_to_prob = get_mean_cell_probs(prediction)

    # Select the answers above a classification threshold.
    answer_coordinates = []
    for col in range(max_width):
      for row in range(max_height):
        cell_prob = cell_coords_to_prob.get((col, row), None)
        if cell_prob is not None:
          if cell_prob > cell_classification_threshold:
            answer_coordinates.append(str((row, col)))

    try:
      example_id, annotator, position = text_utils.parse_question_id(
          question_id)
      position = str(position)
    except ValueError:
      example_id = "_"
      annotator = "_"
      position = "_"
    prediction_to_write = {
        "question_id": question_id,
        "id": example_id,
        "annotator": annotator,
        "position": position,
        "answer_coordinates": str(answer_coordinates)
    }
    if do_model_aggregation:
      prediction_to_write["gold_aggr"] = str(prediction["gold_aggr"][0])
      prediction_to_write["pred_aggr"] = str(prediction["pred_aggr"])
    if do_model_classification:
      prediction_to_write["gold_cls"] = str(prediction["gold_cls"][0])
      prediction_to_write["pred_cls"] = str(prediction["pred_cls"])
      prediction_to_write["logits_cls"] = str(prediction["logits_cls"])
    result_predicted.append(prediction_to_write)
  return result_predicted
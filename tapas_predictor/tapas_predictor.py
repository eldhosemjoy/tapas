import numpy as np
import os,sys
qaa_path = os.getcwd()+"/tapas_predictor/"
sys.path.append(qaa_path)

import json
from tapas.experiments import prediction_utils as exp_prediction_utils
from tapas.models import tapas_classifier_model
from tapas.models.bert import modeling
import tensorflow.compat.v1 as tf

from tapas.utils import tf_example_utils
from tapas.protos import interaction_pb2
from tapas.utils import number_annotation_utils

from tapas.utils import text_utils

from tapas.scripts import prediction_utils

import collections

class TapasPredictor(object):
  """TAPAS predictor class."""

  def __init__(self,config="config.json"):
    """Constructs a Predictor Object."""
    self.config           = self.load_config(config)
    self.session          = self.load_graph()
  
  def load_config(self,config_path):
    config = {}
    with open(config_path) as f:
        config = json.load(f)
    return config
        
  def create_interactions(self, table_data, queries):
    max_seq_length = 512
    config = tf_example_utils.ClassifierConversionConfig(
        vocab_file=self.config['vocab'],
        max_seq_length=max_seq_length,
        max_column_id=max_seq_length,
        max_row_id=max_seq_length,
        strip_column_names=False,
        add_aggregation_candidates=False,
    )
    converter = tf_example_utils.ToClassifierTensorflowExample(config)

    def convert_interactions_to_examples(tables_and_queries):
      """Calls Tapas converter to convert interaction to example."""
      for idx, (table, queries) in enumerate(tables_and_queries):
        interaction = interaction_pb2.Interaction()
        for position, query in enumerate(queries):
          question = interaction.questions.add()
          question.original_text = query
          question.id = f"{idx}-0_{position}"
        for header in table[0]:
          interaction.table.columns.add().text = header
        for line in table[1:]:
          row = interaction.table.rows.add()
          for cell in line:
            row.cells.add().text = cell
        number_annotation_utils.add_numeric_values(interaction)
        for i in range(len(interaction.questions)):
          try:
            yield converter.convert(interaction, i)
          except ValueError as e:
            print(f"Can't convert interaction: {interaction.id} error: {e}")

#     table = [list(map(lambda s: s.strip(), row.split("|"))) 
#             for row in table_data.split("\n") if row.strip()]

    examples = convert_interactions_to_examples([(table_data, queries)])
    return examples, table_data

  def load_frozen_graph(self,frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="tapas")
    return graph


  def load_graph(self):
    graph = self.load_frozen_graph(self.config['model'])
    # loss/mul_5,column_ids,row_ids,segment_ids,question_id_ints
    self.probabilities           = graph.get_tensor_by_name("tapas/loss/mul_5:0")
    self.column_ids              = graph.get_tensor_by_name("tapas/column_ids:0")
    self.row_ids                 = graph.get_tensor_by_name("tapas/row_ids:0")
    self.segment_ids             = graph.get_tensor_by_name("tapas/segment_ids:0")
    self.question_id_ints        = graph.get_tensor_by_name("tapas/question_id_ints:0")

    self.features           = {}
    self.features_list      = ["input_ids","input_mask","column_ids","row_ids","segment_ids","column_ranks","inv_column_ranks","numeric_relations","label_ids","prev_label_ids","question_id","question_id_ints","aggregation_function_id","classification_class_index"]
    for each_feature in self.features_list:
      self.features[each_feature] = graph.get_tensor_by_name("tapas/"+each_feature+":0")
    if(self.config['gpu']):
        gpu_options           = tf.GPUOptions(per_process_gpu_memory_fraction=0.133)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        persistent_sess = tf.Session(graph=graph, config=sess_config)
    else:
        session_conf = tf.ConfigProto(device_count={'CPU' : 1, 'GPU' : 0})
        persistent_sess = tf.Session(graph=graph, config=session_conf)
    return persistent_sess

  def predict(self, table, queries):
    examples, table_parsed = self.create_interactions(table, queries)
    examples_by_position = collections.defaultdict(dict)
    for example in examples:
      question_id = example["question_id"][0, 0].decode("utf-8")
#       print(question_id)
      table_id, annotator, position = text_utils.parse_question_id(question_id)
#       print("Position : ", position)
      example_id = (table_id, annotator)
      examples_by_position[position][example_id] = example
    
    examples = examples_by_position[0]
    current_answer  = {}
    feed_dictionary = {}
    current_scope = ['column_ids','row_ids','segment_ids','question_id_ints','question_id']
    for example in examples.values():
      for feat in self.features_list:
        feed_dictionary[self.features[feat]] = example[feat]
        if(feat in current_scope):
          current_answer[feat]= example[feat]
    probabilities, _, _, _, _ = self.session.run([self.probabilities, self.column_ids, self.row_ids, self.segment_ids, self.question_id_ints],feed_dict=feed_dictionary)

    current_answer['probabilities'] = probabilities

    result_evaluated = {}
    for each_key in current_answer:
      result_evaluated[each_key] = np.concatenate(current_answer[each_key],axis=0)

    _CELL_CLASSIFICATION_THRESHOLD = 0.5
    rowcord = (exp_prediction_utils.retrieve_predictions(
        [result_evaluated],
        False,
        do_model_classification=False,
        cell_classification_threshold=_CELL_CLASSIFICATION_THRESHOLD,
    ))
    coordinates = prediction_utils.parse_coordinates(rowcord[1]["answer_coordinates"])
    prediction = ', '.join([table_parsed[row + 1][col] for row, col in coordinates])
    return prediction

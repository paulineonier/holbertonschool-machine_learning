#!/usr/bin/env python3
"""Question Answering with BERT and TensorFlow Hub."""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """Finds a snippet of text within a reference document to answer a question.

    Args:
        question (str): The question to answer.
        reference (str): The document containing the answer context.

    Returns:
        str: The extracted answer snippet, or None if no valid answer is found.
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

    # Tokenisation et conversion en IDs
    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    # Formatage BERT : [CLS] question [SEP] reference [SEP]
    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + reference_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    
    # Segment IDs (0 pour le symbole CLS + question + SEP, 1 pour reference + SEP)
    type_cls_q = len(question_tokens) + 2  # [CLS] + question + [SEP]
    type_ref = len(reference_tokens) + 1    # reference + [SEP]
    segment_ids = [0] * type_cls_q + [1] * type_ref

    # Conversion en Tensors avec ajout de la dimension batch (shape: [1, seq_len])
    input_ids_tensor = tf.constant([input_ids], dtype=tf.int32)
    input_mask_tensor = tf.constant([input_mask], dtype=tf.int32)
    segment_ids_tensor = tf.constant([segment_ids], dtype=tf.int32)

    # Inférence avec le modèle TF Hub
    outputs = model([input_ids_tensor, input_mask_tensor, segment_ids_tensor])
    start_logits = outputs[0]
    end_logits = outputs[1]

    # Récupération des index de début et de fin avec les plus forts scores (logits)
    start_index = tf.argmax(start_logits, axis=-1).numpy()[0]
    end_index = tf.argmax(end_logits, axis=-1).numpy()[0]

    # Si l'index de fin précède l'index de début, aucune réponse valide n'a été trouvée
    if start_index > end_index:
        return None

    # Extraction des tokens correspondant à la réponse et reconstruction de la chaîne
    answer_tokens = tokens[start_index:end_index + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer if answer.strip() else None
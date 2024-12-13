import re
import torch
import numpy as np

RE_TAGS = {
  'cv':'changing value',
  'cm':'command',
  'dv':'device',
  'dr':'duration',
  'loc':'location',
  'sc':'scene',
  'tn':'target number',
  'ta':'time at'
}

def extract_entities(tokens, labels):
    entities = []
    current_entity = None

    for token, label in zip(tokens, labels):
        label_type = label.split("-")[1] if "-" in label else None
        if label_type:
            label_type = RE_TAGS[label_type]
            if current_entity and label_type != current_entity["type"]:
                entities.append(current_entity)
                current_entity = None

            if not current_entity:
                current_entity = {"type": label_type, "filler": token}
            else:
                current_entity["filler"] += " " + token
        elif current_entity:
            entities.append(current_entity)
            current_entity = None
    if current_entity:
        entities.append(current_entity)

    return entities

def split_char(text_sentence):
    words = re.findall(r'\w+|\S', text_sentence)
    return words

def predict_ner(model_ner,text_sentence, tokenizer, tag_values):
  tokenized_sentence = tokenizer.encode(text_sentence)
  input_ids = torch.tensor([tokenized_sentence])
  with torch.no_grad():
      output = model_ner(input_ids)
  label_indices = np.argmax(output[0].to('cpu').numpy(), axis=1)
  tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
  new_tokens, new_labels = [], []
  for token, label_idx in zip(tokens, label_indices):
      if token.startswith("##"):
          new_tokens[-1] = new_tokens[-1] + token[2:]
      else:
          new_labels.append(tag_values[label_idx])
          new_tokens.append(token)
  new_labels.pop(0)
  new_labels.pop(-1)
  new_tokens = split_char(text_sentence)
  result = extract_entities(new_tokens, new_labels)
  return result
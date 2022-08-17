from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from scipy.special import softmax
import numpy as np
import sys
import json
import torch
import gzip

BATCHSIZE = 512
MAX_SEQ_LEN = 128 # more than enough for over 99% of our tweets
MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
DEVICE = torch.device("cuda")

# Taken from https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment
# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def model_predict(model, tokenizer, texts):
    texts = [preprocess(text) for text in texts]

    encoded_input = tokenizer(texts, return_tensors='pt', padding=True,
                              truncation=True, max_length=MAX_SEQ_LEN)
    output = model(**encoded_input)

    scores = output.logits.detach().cpu().numpy()
    scores = softmax(scores, axis=-1).tolist()
    return scores

if __name__ == "__main__":
    # Inputs
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    # Load model
    config = AutoConfig.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model = torch.nn.DataParallel(model)
    model.to(DEVICE)
    model.eval()

    # Open the output file
    output_file = open(output_filename, "w", encoding="utf-8")

    # Open and process the input file
    with gzip.open(input_filename, "rt", encoding="utf-8") as input_file:

        curr_batch = []
        for i, line in enumerate(input_file):
            line = json.loads(line)
            twt_id_str = list(line.keys())[0]
            twt_txt = line[twt_id_str]
            curr_batch.append({"twt_id_str": twt_id_str, "twt_txt": twt_txt})

            if len(curr_batch) == BATCHSIZE:
                texts = [d["twt_txt"] for d in curr_batch]
                scores = model_predict(model, tokenizer, texts)
                for j, score in enumerate(scores):
                    curr_d = curr_batch[j]
                    curr_d["sentiment_scores"] = score
                    curr_d["final_sentiment"] = config.id2label[np.argmax(score)]
                    output_file.write(json.dumps(curr_d) + "\n")

                curr_batch = []

    if len(curr_batch) != 0:
        texts = [d["twt_txt"] for d in curr_batch]
        scores = model_predict(model, tokenizer, texts)
        for j, score in enumerate(scores):
            curr_d = curr_batch[j]
            curr_d["sentiment_scores"] = score
            curr_d["final_sentiment"] = config.id2label[np.argmax(score)]
            output_file.write(json.dumps(curr_d) + "\n")


    output_file.close()

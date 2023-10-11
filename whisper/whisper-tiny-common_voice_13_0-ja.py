from datasets import load_dataset, DatasetDict


dataset_id = "CWKSC/common_voice_13_0-ja-whisper-tiny" # "mozilla-foundation/common_voice_11_0"
model_id = "openai/whisper-tiny"
model_language = "japanese"
push_to_hub_id = "whisper-tiny-common_voice_13_0-ja"

common_voice = load_dataset(dataset_id)
print(common_voice)


from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
tokenizer = WhisperTokenizer.from_pretrained(model_id, language=model_language, task="transcribe")
processor = WhisperProcessor.from_pretrained(model_id, language=model_language, task="transcribe")



import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)



import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}



from transformers import WhisperForConditionalGeneration, AutoModelForSpeechSeq2Seq
from peft import prepare_model_for_int8_training, LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained(model_id, device_map="auto")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []   


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=f"./{push_to_hub_id}",  # change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    warmup_steps=50,
    num_train_epochs=1,
    evaluation_strategy="no", # "epoch",
    fp16=True,
    # per_device_eval_batch_size=4,
    generation_max_length=128,
    logging_steps=200,
    remove_unused_columns=False, # https://discuss.huggingface.co/t/indexerror-invalid-key-16-is-out-of-bounds-for-size-0/14298/24
    label_names=["labels"],
    eval_accumulation_steps = 1
)


from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    # eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)


trainer.train()


model.push_to_hub(push_to_hub_id)
feature_extractor.push_to_hub(push_to_hub_id)
tokenizer.push_to_hub(push_to_hub_id)
processor.push_to_hub(push_to_hub_id)


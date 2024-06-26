from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

training_args = TrainingArguments(
    output_dir="path/to/save/folder/",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

dataset = load_dataset("rotten_tomatoes")  # doctest: +IGNORE_RESULT

def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])

dataset = dataset.map(tokenize_dataset, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)  # doctest: +SKIP

trainer.train()
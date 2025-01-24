from transformers import BlenderbotForConditionalGeneration, BlenderbotSmallTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Load the BlenderBot-small model and tokenizer
model_name = "facebook/blenderbot_small-90M"
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_name)

# Load your dataset
import json
with open("dataset.json", "r") as f:
    data = json.load(f)

# Convert to Hugging Face Dataset
dataset = Dataset.from_dict({"input": [d["input"] for d in data], "output": [d["output"] for d in data]})

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=128)
    outputs = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=128)
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": outputs["input_ids"]}

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir="./fine-tuned-blenderbot",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-blenderbot")
tokenizer.save_pretrained("./fine-tuned-blenderbot")
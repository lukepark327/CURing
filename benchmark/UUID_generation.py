import uuid
from datasets import Dataset, DatasetDict
import random

num_pairs = 1024  # TODO
raw_dataset_path = "./benchmark/uuid_raw_dataset"

# Generate the UUID dataset
uuid_pairs = []
for _ in range(num_pairs):
    input_uuid = str(uuid.uuid4())
    output_uuid = str(uuid.uuid4())
    # We store the input prompt and the expected answer separately
    input_text = f"Given this UUID: {input_uuid}\nThe corresponding UUID is: "
    uuid_pairs.append({"input_text": input_text, "output_uuid": output_uuid})

# Shuffle the data
random.shuffle(uuid_pairs)

# Split into train and validation
train_data = uuid_pairs[:]
val_data = uuid_pairs[:]

dataset = DatasetDict({
    'train': Dataset.from_dict({
        "input_text": [d["input_text"] for d in train_data],
        "output_uuid": [d["output_uuid"] for d in train_data]
    }),
    'validation': Dataset.from_dict({
        "input_text": [d["input_text"] for d in val_data],
        "output_uuid": [d["output_uuid"] for d in val_data]
    })
})

dataset.save_to_disk(raw_dataset_path)
print("Raw UUID mapping dataset generated and saved successfully.")

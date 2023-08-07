# esm2_LoRA_binding_sites
Training a Low Rank Adaptation of the protein language model ESM-2 for RNA binding site predictor

## Training the Model
To train your own, clone the repo and create a conda environment using the `environment.yaml` file. Then run:
```
from lora_binding_sites_newest_v3 import main

main()
```

## Using the model:
To use the model, try running:
```
from transformers import AutoModelForTokenClassification, AutoTokenizer
from peft import PeftModel
import torch

# Path to the saved LoRA model
model_path = "best_model_dir\\final_best_model"
# ESM2 base model
base_model_path = "facebook/esm2_t6_8M_UR50D"

# Load the model
base_model = AutoModelForTokenClassification.from_pretrained(base_model_path)
loaded_model = PeftModel.from_pretrained(base_model, model_path)

# Ensure the model is in evaluation mode
loaded_model.eval()

# Load the tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Protein sequence for inference
protein_sequence = "MAVPETRPNHTIYINNLNEKIKKDELKKSLHAIFSRFGQILDILVSRSLKMRGQAFVIFKEVSSATNALRSMQGFPFYDKPMRIQYAKTDSDIIAKMKGT"  # Replace with your actual sequence

# Tokenize the sequence
inputs = loaded_tokenizer(protein_sequence, return_tensors="pt", truncation=True, max_length=1024, padding='max_length')

# Run the model
with torch.no_grad():
    logits = loaded_model(**inputs).logits

# Get predictions
tokens = loaded_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  # Convert input ids back to tokens
predictions = torch.argmax(logits, dim=2)

# Define labels
id2label = {
    0: "No binding site",
    1: "Binding site"
}

# Print the predicted labels for each token
for token, prediction in zip(tokens, predictions[0].numpy()):
    if token not in ['<pad>', '<cls>', '<eos>']:
        print((token, id2label[prediction]))
```

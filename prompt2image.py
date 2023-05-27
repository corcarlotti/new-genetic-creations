import argparse
from diffusers import StableDiffusionPipeline
import torch

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run a prompt with the stable diffusion model on trained embedding.')
parser.add_argument('--prompt', type=str, help='Input prompt for the script')
args = parser.parse_args()

# Now, args.prompt will contain the input prompt passed on the command line
prompt = args.prompt

# ... rest of your script using the prompt ...
# 2 step: take the returning input string and run the model + embedding with it

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to("cpu")

pipe.load_textual_inversion("/Users/roschkach/Projekte/NewGeneticCreations/models/dna2image_2v.pt")

prompt = "dna2image_2v-25000,"+prompt.upper()
image = pipe(prompt, num_inference_steps=100).images[0]
image.save("/Users/roschkach/Projekte/NewGeneticCreations/output_first_prototype/test_creation_3.png")
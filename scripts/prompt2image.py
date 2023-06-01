import argparse
from diffusers import StableDiffusionPipeline
import torch

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run a prompt with the stable diffusion model on trained embedding.')
parser.add_argument('--prompt', type=str, default='', help='Input prompt for the script')
parser.add_argument('--inf_steps', type=int, default=100, help='Number of inference steps')
parser.add_argument('--embedding', type=str, default="/workspace/new-genetic-creations/embeddings/dna2image_2v.pt", help='Path to embedding')
parser.add_argument('--image_folder', type=str, default="/workspace/new-genetic-creations/images/", help='Path to folder where images are saved')
parser.add_argument('--image_name', type=str, default='genetic_creation.png', help='Name for image')
args = parser.parse_args()

# Now, args.prompt will contain the input prompt passed on the command line
prompt = args.prompt
inf_steps = args.inf_steps
path_emb = args.embedding
image_folder = args.image_folder
image_name = args.image_name

# take the input string and run the model + embedding with it

# load model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch.float16).to("cuda")

# load embedding
pipe.load_textual_inversion(path_emb)

# run prompt
prompt = "dna2image_2v-25000,"+prompt.upper()
image = pipe(prompt, num_inference_steps=inf_steps).images[0]
image.save(image_folder + image_name)
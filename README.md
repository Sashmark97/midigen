# midigen
Test task for MIDI files generatioin
## Preparation Guide
1. Clone the repository
2. Install modules from requirements.txt
3. Download pre-trained models with script download_models.sh from Saved_models folder

##Training guide
1. Choose one of experiments config from Experiments folder or create your own (instructions on these are yet to come)
2. Run ```python train.py --yaml_path='/path/to/experiment.yml''``` in scripts folder

## Generation guide
1. Choose one of the models, downloaded with script or trained yourself
2. Run ```python generate.py --model_type='GPT' --model_path='/path/to/saved/model' --output_dir='/dir/to/output/midi' --dataset_pickle='/path/to/data/split.pkl' --number_of_seqs=number_of_files_to_generate --max_seq=max_primer_len --target_len=output_len --beam=beam_search_number --device='cuda:3'```

* If beam is equal to 0 no beam search is used, instead we use regular categorial distribution from token probabilities
* Number of seqs is number of sequences used as primer to generate MIDI

! Using this library you may encounter torch.device problems. I suggest you change value of TORCH_CUDA_DEVICE in midigen.utils.constants to 
your used device or even disable CUDA by setting USE_CUDA to 0. Later on I will implement constant-agnostic device choice
to prevent such errors

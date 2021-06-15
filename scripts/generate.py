import os
import sys
import yaml
import pickle
import shutil
import  argparse
from torch.utils.data import DataLoader
sys.path.append('..')

from midigen.models.gpt2.gpt2 import GPT
from midigen.data.dataset import EPianoDataset
from midigen.data.neural_processor import encode_midi, decode_midi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='GPT', help="Type of model: GPT or Transformer")
    parser.add_argument('--model_path', help="Path to model weights")
    parser.add_argument('--output_dir', help="Path to output results")
    parser.add_argument('--dataset_pickle', help="Path to model weights")
    parser.add_argument('--number_of_seqs', default=10, type=int, help='Number of primer sequences to generate from')
    parser.add_argument('--max_seq', default=512, type=int, help='Length of primer sequence')
    parser.add_argument('--target_len', default=2048, type=int, help="Length of model output")
    parser.add_argument('--beam', default=0, type=int, help="Number of hypothesis in beam search")
    parser.add_argument('--device', default='cpu', help="Path to model weights")
    args = parser.parse_args()


    if args.model_type == 'GPT':
        model = GPT.load(args.model_path, device=args.device).to(args.device).eval()
    else:
        raise NotImplementedError(f'Model {args.model_type} not implemented yet.')
    with open(args.dataset_pickle, 'rb') as f:
        data = pickle.load(f)

    test_dataset = EPianoDataset(data['test'], max_seq=args.max_seq, random_seq=False,
                                 num_files=5, type='test')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f'Saving midi files to {args.output_dir} with beam of {args.beam}')
    for idx in range(args.number_of_seqs):
        primer, _ = test_dataset[idx]
        primer = primer.to(args.device)
        decode_midi(primer[:args.max_seq].cpu().numpy(), file_path=os.path.join(args.output_dir, f"primer_{idx}.mid"))
        rand_seq = model.generate(primer[:args.max_seq], args.target_len, beam=args.beam)
        decode_midi(rand_seq[0].cpu().numpy(), file_path=os.path.join(args.output_dir, f"generated_beam{args.beam}_{idx}.mid"))

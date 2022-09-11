from train import Model, Dataset
import argparse
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.pt')
    parser.add_argument('--input-dir', type=str, default='./data/anekdoty.txt')
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--length', type=int, default=10)
    args = parser.parse_args()

    dataset = Dataset(args.input_dir)

    vocab_size = torch.load(args.model, map_location='cpu')['embedding.weight'].shape[0]
    model = Model(vocab_size, args.model)

    print(' '.join(model.generate(dataset, args.prefix, next_words=args.length)))
import argparse
from model import train_model



def main():
    parser = argparse.ArgumentParser(description='?')
    parser = argparse.ArgumentParser(description='Train a new network on a dataset')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--data_dir', type=str, help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default='new_model_checkpoint.pth', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str,  help='Choose architecture (e.g., "vgg13_25088_hidden_unit, alexnet_9216")')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    

    args = parser.parse_args()
    
    train_model(gpu=args.gpu, model_name=args.arch, hidden_units=args.hidden_units, data_dir=args.data_dir, save_dir=args.save_dir,          epochs=args.epochs, learning_rate=args.learning_rate)


if __name__ == '__main__':
    main()

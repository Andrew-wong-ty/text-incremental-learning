import argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')


parser.add_argument('--seed', type=int, default=16, help='')
parser.add_argument('--print_freq', type=int, default=100, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')

parser.add_argument('--cuda_index', type=int, default=1, help='')
parser.add_argument('--base_class', type=int, default=5, help='')
parser.add_argument('--phase', type=int, default=5, help='')

parser.add_argument('--dataset', type=str, default="Wiki80", help='')
parser.add_argument('--model_save_path', type=str, default="/data/tywang/Increment_learning", help='')
parser.add_argument('--model', type=str, default="/data/transformers/bert-base-cased", help='')
parser.add_argument('--momentum', type=float, default=0.9, help='')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='')
parser.add_argument('--lr', type=float, default=6e-5, help='')
parser.add_argument('--start_epoch', type=int, default=0, help='')
parser.add_argument('--epochs', type=int, default=8, help='')
parser.add_argument('--max_length', type=int, default=128, help='')
parser.add_argument('--Hidden', type=int, default=5120, help='')
parser.add_argument('--rg', default=1e-1, type=float,help='')
parser.add_argument('--repeat', default=1, type=int,help='')
args = parser.parse_args()
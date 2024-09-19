import argparse
import torch
args = argparse.Namespace()

def parse_arguments():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference-only', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--autocast', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--use-ckpt', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--bf16', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--max-epochs', default=5, type=int)
    parser.add_argument('--max-steps', default=100000, type=int)
    parser.add_argument('--num-workers', default=15, type=int)
    parser.add_argument('--deterministic', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--train-only', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--pad', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--num-buckets', default=0, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', choices=['hpu', 'cpu', 'cuda'], default='hpu', type=str)
    parser.add_argument('--ckpt-path', default="./cppe-5.ckpt", type=str)
    parser.add_argument('--ckpt-store-interval-epochs', default=0, type=int)
    parser.add_argument('--ckpt-store-path', default="./", type=str)
    
    # Arguments specifically for inference
    parser.add_argument('--max-inf-frames', default=0, type=int)
    parser.parse_args(namespace=args)
    if args.deterministic:
        args.num_workers = 0

def show_arguments():
    print(f'inference-only = {args.inference_only}')
    print(f'autocast = {args.autocast}')
    print(f'use-ckpt = {args.use_ckpt}')
    print(f'bf16 = {args.bf16}')
    print(f'batch-size = {args.batch_size}')
    print(f'max-epochs = {args.max_epochs}')
    print(f'max-steps = {args.max_steps}')
    print(f'num-workers = {args.num_workers}')
    print(f'deterministic = {args.deterministic}')
    print(f'train-only = {args.train_only}')
    print(f'pad = {args.pad}')
    print(f'num-buckets = {args.num_buckets}')
    print(f'threshold = {args.threshold}')
    print(f'seed = {args.seed}')
    print(f'device = {args.device}')
    print(f'ckpt-path = {args.ckpt_path}')
    print(f'ckpt-store-interval-epochs = {args.ckpt_store_interval_epochs}')
    print(f'ckpt-store-path = {args.ckpt_store_path}')
    print(f'max-inf-frames = {args.max_inf_frames}')
    
    # Derived parameters
    precision = torch.bfloat16 if args.bf16 else torch.float32
    shuffle = False if args.deterministic else True
    print(f'Derived params: precision = {precision}, shuffle = {shuffle}')
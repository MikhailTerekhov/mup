
import torch as t
import os
import argparse
from trainer import TransformerTrainer, prepare_data
from modules import DemoTransformer
from dotenv import load_dotenv
from params import Config, TransformerTrainingArgs

# Set the environment variable for WANDB to silent mode
os.environ["WANDB_SILENT"] = "true"

def main(args):
    # Initialize configuration for the transformer model
    load_dotenv()
    device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')

    dataset = prepare_data(max_length=args.max_length)

    model_cfg = Config(
        debug=False,
        apply_muP=args.apply_muP,
        d_head=args.d_head,
        d_model=args.width,
        d_mlp=args.width * 4,
        n_heads=int(args.width / args.d_head),
        n_layers=args.n_layers,
        n_ctx=args.max_length,
        mu_output_alpha=args.mu_output_alpha,
        mu_input_alpha=args.mu_input_alpha,
        muP_width_multiplier=args.width / args.base_width,
        init_std=args.init_std,
        d_vocab=50257  # GPT2 vocab size
    )

    # Prepare training arguments
    training_args = TransformerTrainingArgs(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        batch_size=args.batch_size,
        max_steps_per_epoch=args.max_steps_per_epoch,
        wandb_project=args.wandb_project,
        wandb_name=f"{args.wandb_run_prefix}_width_{args.width}_{'muP' if args.apply_muP else 'std'}_lr_{args.lr}",
        collect_norms=args.collect_norms,
    )

    # Instantiate the model and the trainer
    model = DemoTransformer(model_cfg).to(device)
    trainer = TransformerTrainer(training_args, model, dataset)

    # Start training
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model with configurable settings via command line.")

    parser.add_argument("--apply_muP", action="store_true", help="Apply muP modifications or not.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--max_steps_per_epoch", type=int, default=-1, help="Maximum steps per epoch.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--init_std", type=float, default=0.02, help="Initial standard deviation for weights.")
    parser.add_argument("--mu_input_alpha", type=float, default=1.0, help="muP input alpha.")
    parser.add_argument("--mu_output_alpha", type=float, default=1.0, help="muP output alpha.")
    parser.add_argument("--d_head", type=int, default=64, help="Model dimension size per head.")
    parser.add_argument("--width", type=int, default=256, help="Width of the model.")
    parser.add_argument("--base_width", type=int, default=256, help="Base width for muP width multiplier.")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers in the model.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.99, help="Beta2 for Adam optimizer.")
    parser.add_argument("--collect_norms", action="store_true", help="Whether to collect layer norms data.")
    parser.add_argument("--wandb_project", type=str, default="mup-transformer-training", help="The W&B project name.")
    parser.add_argument("--wandb_run_prefix", type=str, default="transformer_traintest", help="Prefix for the W&B run name.")

    args = parser.parse_args()

    main(args)




if __name__ == "__main__":
    
    dataset = prepare_data(max_length=1024)
    pass
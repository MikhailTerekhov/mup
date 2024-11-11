import torch as t
import wandb
from params import TransformerTrainingArgs
from modules import DemoTransformer
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate
from transformers import GPT2Tokenizer
from tqdm import tqdm
import numpy as np

device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')

def prepare_data(max_length):
    dataset = load_dataset("Salesforce/wikitext","wikitext-2-raw-v1")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset["train"] = tokenize_and_concatenate(dataset["train"], tokenizer, streaming=False, max_length=max_length, column_name="text", add_bos_token=True, num_proc=4)
    dataset["validation"] = tokenize_and_concatenate(dataset["validation"], tokenizer, streaming=False, max_length=max_length, column_name="text", add_bos_token=True, num_proc=4)
    return dataset


def norm_logging_hook(log_key, trainer):
    def hook(module, input, output):
        #norm = np.output.norm(dim=-1).mean()
        norm = t.mean(t.abs(output))
        wandb.log({log_key: norm}, step=trainer.step)
    return hook

def get_log_probs(logits,tokens):

    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    return log_probs_for_tokens

class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model: DemoTransformer, dataset_dict):
        super().__init__()
        self.model = model
        self.args = args
        self.dataset_dict = dataset_dict
        
        self.optimizer = model.configure_optimizers(args.weight_decay, args.lr, args.betas)
        self.step = 0
        if args.collect_norms:
            self.register_hooks()

    def register_hooks(self):
        self.model.embed.register_forward_hook(norm_logging_hook('embedding_norm', self))

        for i, block in enumerate(self.model.blocks):
            block.attn.register_forward_hook(norm_logging_hook(f'attn_out_norm_block_{i}', self))
            block.mlp.register_forward_hook(norm_logging_hook(f'mlp_out_norm_block_{i}', self))

        self.model.ln_final.register_forward_hook(norm_logging_hook('ln_final_norm', self))
        self.model.unembed.register_forward_hook(norm_logging_hook('unembedding_norm', self))

    def training_step(self, batch):
        '''
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        '''
        tokens = batch["tokens"].to(device)
        logits = self.model(tokens)
        loss = -get_log_probs(logits, tokens).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        wandb.log({"train_loss": loss}, step=self.step)
        return loss


    def validation_step(self, batch):
        '''
        Calculates & returns the accuracy on the tokens in the batch (i.e. how often the model's prediction
        is correct). Logging should happen in the `train` function (after we've computed the accuracy for
        the whole validation set).
        '''
        with t.no_grad():
            tokens = batch["tokens"].to(device)

            logits = self.model(tokens)
            loss = -get_log_probs(logits, tokens).mean()

            last_logits = logits[:, :-1]
            predicted_tokens = last_logits.argmax(dim=-1)
            correct_predictions = (predicted_tokens == tokens[:, 1:]).flatten()
        
        return (correct_predictions, loss)

    
    def train(self):
        '''
        Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
        for each epoch at `self.args.max_steps_per_epoch` steps.
        '''
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)
        accuracy = np.nan


        progress_bar = tqdm(total = self.args.epochs)

        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader()):
                loss = self.training_step(batch)
                
                progress_bar.set_description(f"Epoch {epoch+1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}")
                if self.args.max_steps_per_epoch>0 and i >= self.args.max_steps_per_epoch:
                    break
            val_results = [self.validation_step(batch) for batch in self.val_loader()]
            progress_bar.update()
            correct_predictions = t.concat([batch_res[0]for batch_res in val_results])
            losses = t.concat([batch_res[1].reshape(1) for batch_res in val_results])
            accuracy = correct_predictions.float().mean().item()
            val_loss = losses.mean().item()
            wandb.log({"val_accuracy": accuracy, "val_loss": val_loss}, step=self.step)

        wandb.finish()


    def train_loader(self) -> DataLoader:
        '''Returns train loader (as in code above).'''
        return DataLoader(self.dataset_dict["train"], batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    def val_loader(self) -> DataLoader:
        '''Returns validation loader (as in code above).'''
        return DataLoader(self.dataset_dict["validation"], batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_loader(self) -> DataLoader:
        '''Returns test loader (as in code above).'''
        return DataLoader(self.dataset_dict["test"], batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
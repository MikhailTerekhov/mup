# muP-autoregressive
This notebook contains the muP implementation suggested [in this EleutherAI blog](https://blog.eleuther.ai/mutransfer/).
Base transformer implementation is based on GPT2, provided [in this Arena exercise](https://arena3-chapter1-transformer-interp.streamlit.app/[1.1]_Transformer_from_Scratch).

In the [modules.py](https://github.com/BerkeArgin/muP-autoregressive/blob/main/modules.py), you can search for `cfg.apply_muP` string to see the changes between standard and muP implementation.

## Usage
1. Run `pip install -r requirements.txt`
2. Create an `.env` file with fields `HF_TOKEN`, `WANDB_API_KEY` containing HuggingFace access token and W&B API keys.
3. To conduct coordinate check test, run `./run_grid_coord.sh`
4. To start training over a grid of widths and learning rates, run `./run_grid_train.sh`
5. To visualize W&B logs, check [notebook.ipynb](https://github.com/BerkeArgin/muP-autoregressive/blob/main/notebook.ipynb)

To train and experiment with different configurations, check [train.py](https://github.com/BerkeArgin/muP-autoregressive/blob/main/train.py)



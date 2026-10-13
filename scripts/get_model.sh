#!/bin/sh
# Downloads the TinyStories 15M checkpoint and tokenizer into models/.
set -e
mkdir -p models
curl -L -o models/stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
curl -L -o models/tokenizer.bin https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin
echo "done: models/stories15M.bin and models/tokenizer.bin"

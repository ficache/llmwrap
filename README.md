
![alt text](https://github.com/Dixxe/LLMwrap/blob/main/llmwrap-head.png?raw=true)

# LLMwrapp

A wrapper for LLMs written fully in rust as hobby project to practice my rust skills and knowledge!

## Supported models

LLMwrap based on rustformers, but in current state only **llama** or **llama based** models are supported. I succsesfully run and test this project with [Vicuna](https://huggingface.co/TheBloke/vicuna-7B-v1.5-GGML)

## Roadmap

[x] Refactor all main.rs to struct and functions

- Use dotenv for model loading

- Implement async functions for anwser generation

- Add support for all models that supported by `rustformers` crate

## Installation

Install `llmwrap` via git

```bash
  git clone https://github.com/Dixxe/llmwrap.git
  cd llmwrap
```
Now install any Llama **GGMLv3** language model and put it inside folder.

edit line of code that defines path to model

`let model = load_model("your_path_here", 2048, true, Some(10));`

Change code related to personality, name and your prompts!

Run everything with

`cargo run`

(WIP: all variables will be stored in .env in future)    

## Contributing

Contributions are always welcome!

## Dependencies

This project uses the following dependencies:

- [rustformers/llm](https://github.com/rustformers/llm) - Licensed under both the Apache License 2.0 and the MIT License.


## Authors

- [@Dixxe](https://github.com/Dixxe)


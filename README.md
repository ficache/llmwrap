
![Logo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/th5xamgrr6se0x5ro4g6.png)


# LLMwrapp

A wrapper written in rust, that gives you simple control over your LLM models and their parametrs!

Currently is heavy WIP.

## Roadmap

- Refactor all main.rs to struct and functions

- Use dotenv for model loading

- Implement async functions and streaming output of the model

- Add support for all models that supported by `rustformers` crate

## Installation

Install `llmwrap` via git

```bash
  git clone https://github.com/Dixxe/llmwrap.git
  cd llmwrap
```
Now install any Llama **GGMLv3** language model and put it inside folder.

edit line of code that defines path to model

(WIP: will be change to .env variable in future)    

## Contributing

Contributions are always welcome!

## Dependencies

This project uses the following dependencies:

- [rustformers/llm](https://github.com/rustformers/llm) - Licensed under both the Apache License 2.0 and the MIT License.


## Authors

- [@Dixxe](https://github.com/Dixxe)


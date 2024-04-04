This is an edited version of VLLMs. The changes include:

- In streaming mode, returning each token text instead of the whole sequence. Good for integrated with Huggingface Chat UI.
- Add api for RAGs application where inputs contain message, relevant contexts and history.

Setup:
```
pip install https://github.com/thanhdath/vllm.git
```

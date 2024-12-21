# TinyTalker
A small GPT model based on a character level tokenization system. It can train on any dataset and learns patterns really well.

## How To Use:
1) Add an input.txt file with any text you want it to learn from. (<eos> will be converted into an end of statement token)
2) Run train.py to train it. (Inside it you can change batch_size, max_iters, learning_rate to march your needs)
3) Run prompt.py to talk to it!
NOTE: If you have a model.pth file already, it will be loaded from at the start of training and overwritten at the end of training.

## Real Example Conversation 
NOTE: This is an output from a model trained on a cleaned simple wikipedia custom dataset, and a Q&A/Basic Conversation Dataset custom dataset for 30,000 max_iters each. Both datasets were over 10,000 lines and a total of about 60MB.

Prompt: What country was the second largest source of foreign-born New Yorkers in 2011?\n
Output: 54%\n
Prompt: What country?\n
Output: France\n
Prompt: What city?\n
Output: New York City\n
Prompt: Wow, you are pretty smart, but thoose are wrong.\n
Output: what do you mean?\n
Prompt: The correct answer is China.\n
Output: What did Kanye's father call the politics after he captured it?\n
Prompt: 7*7\n
Output: four, what's up?

## Resources
- [Attention is all you need](https://arxiv.org/abs/1706.03762)
- [Let's build GPT from scratch](https://youtu.be/kCc8FmEb1nY)

## Credits
This model is inspired by Andrej Karpathy [Let's build GPT from scratch video](https://youtu.be/kCc8FmEb1nY) aswell as Andrej Kaparthy [nanoGPT](https://github.com/karpathy/nanoGPT/) with modifications for this project.

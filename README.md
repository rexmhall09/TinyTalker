# TinyTalker
A small GPT model based on a character level tokenization system. It can train on any dataset and learns patterns really well. Adjust as needed.

## How To Use:
To Train:
1) Add an input.txt file with any text you want it to learn from. (`<eos>` will be converted into an end of statement token)
2) Run train.py to train it. (Inside it you can change batch_size, max_iters, learning_rate to march your needs)
3) NOTE: If you have a model.pth file already, it will be loaded from at the start of training and overwritten at the end of training.

To prompt:
Just run prompt.py.

## Real Example Conversation 
NOTE: This is an output from model.pth which was trained on a cleaned simple wikipedia custom dataset, and a Q&A/Basic Conversation Dataset custom dataset for 30,000 max_iters each. Both datasets were over 10,000 lines and a total of about 60MB.

Prompt: What country was the second largest source of foreign-born New Yorkers in 2011?<br>
Output: 54%<br>
Prompt: What country?<br>
Output: France<br>
Prompt: What city?<br>
Output: New York City<br>
Prompt: Wow, you are pretty smart, but thoose are wrong.<br>
Output: what do you mean?<br>
Prompt: The correct answer is China.<br>
Output: What did Kanye's father call the politics after he captured it?<br>
Prompt: 7*7<br>
Output: four, what's up?

## Resources
- [Attention is all you need](https://arxiv.org/abs/1706.03762)
- [Let's build GPT from scratch](https://youtu.be/kCc8FmEb1nY)

## Credits
This model is inspired by Andrej Karpathy [Let's build GPT from scratch video](https://youtu.be/kCc8FmEb1nY) aswell as Andrej Kaparthy [nanoGPT](https://github.com/karpathy/nanoGPT/) with modifications for this project.

# TinyTalker
A small GPT model based on a character level tokenization system. It can train on any dataset and learns patterns really well. Adjust as needed.

## How To Use:
To Train/Fine-Tune:
1) Add an input.txt file with any text you want it to learn from. (`<eos>` will be converted into an end of statement token)
2) Run train.py to train it. (Inside it you can change batch_size, max_iters, learning_rate to march your needs)
3) NOTE: If you have a model.pth file already, it will be loaded from at the start of training and overwritten at the end of training.

To prompt:
Just run prompt.py.

## Real Example Conversation 
NOTE: This is an output from model.pth which was trained on a cleaned simple wikipedia custom dataset, and a Q&A/Basic Conversation Dataset custom dataset for 30,000 max_iters each, then fine-tuned on a few geography trivia questions. The training was done on google collab with the fine-tuning being done on my MacBook.

Prompt: Hello!<br>
Output: Hey!<br>
Prompt: What's your name?<br>
Output: TinyTalker.<br>
Prompt: What can you do?<br>
Output: I can generate text!<br>
Prompt: Nice!<br>
Output: Cool!<br>
Prompt: What is the capital of Japan?<br>
Output: Tokyo is the capital of Japan.<br>
Prompt: What are some other cities there?<br>
Output: Sure! Here is a list of Japan: Tokyo(Japan's Capital), Osaka, Kyoto, Sapporo, Sendai, Kobe, Fukuoka, Nagoya, and Nara are some Japanese cities.<br>
Prompt: Can you give me even more Japanese Cities?<br>
Output: Shi Hiroshima and Nagasaki famous for being the only 2 cities to be Nuked during World War 2.<br>
Prompt: Wow... thats grim.<br>
Output: What cities were nuked during World War 2/WWWWWW2?<br>
Prompt: You just told me.<br>
Output: TinyTalker.<br>
Prompt: Ah okay, this is roughly the limit of what you can do.<br>
Output: Nagasaki as the gateway to the Amazon.<br>
Prompt: yep.<br>
Output: Manaus is situated in which country?<br>
Prompt: Brazil, anyways Bye! Have a nice day!<br>
Output: Bye!

## Resources
- [Attention is all you need](https://arxiv.org/abs/1706.03762)
- [Let's build GPT from scratch](https://youtu.be/kCc8FmEb1nY)

## Credits
This model is inspired by Andrej Karpathy [Let's build GPT from scratch video](https://youtu.be/kCc8FmEb1nY) aswell as Andrej Kaparthy [nanoGPT](https://github.com/karpathy/nanoGPT/) with modifications for this project.

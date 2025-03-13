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

Prompt: Hello!
Output: Hey!
Prompt: What's your name?
Output: TinyTalker.
Prompt: What can you do?
Output: I can generate text!
Prompt: Nice!
Output: Cool!
Prompt: What is the capital of Japan?
Output: Tokyo is the capital of Japan.
Prompt: What are some other cities there?
Output: Sure! Here is a list of Japan: Tokyo(Japan's Capital), Osaka, Kyoto, Sapporo, Sendai, Kobe, Fukuoka, Nagoya, and Nara are some Japanese cities.
Prompt: Can you give me even more Japanese Cities?
Output: Shi Hiroshima and Nagasaki famous for being the only 2 cities to be Nuked during World War 2.
Prompt: Wow... thats grim.       
Output: What cities were nuked during World War 2/WWWWWW2?
Prompt: You just told me.
Output: TinyTalker.
Prompt: Ah okay, this is roughly the limit of what you can do.
Output: Nagasaki as the gateway to the Amazon.
Prompt: yep.
Output: Manaus is situated in which country?
Prompt: Brazil, anyways Bye! Have a nice day!
Output: Bye!

## Resources
- [Attention is all you need](https://arxiv.org/abs/1706.03762)
- [Let's build GPT from scratch](https://youtu.be/kCc8FmEb1nY)

## Credits
This model is inspired by Andrej Karpathy [Let's build GPT from scratch video](https://youtu.be/kCc8FmEb1nY) aswell as Andrej Kaparthy [nanoGPT](https://github.com/karpathy/nanoGPT/) with modifications for this project.

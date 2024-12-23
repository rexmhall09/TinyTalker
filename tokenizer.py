import itertools

UNK_TOKEN = '<unk>'
EOS_TOKEN = '<eos>'
CHARS_FILE = 'chars.txt' 

class Tokenizer:
    def __init__(self, chars_file=CHARS_FILE, unk_token=UNK_TOKEN, eos_token=EOS_TOKEN):
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.chars_file = chars_file
        self._load_chars()
        self._create_mappings()

    def _load_chars(self):
        with open(self.chars_file, 'r', encoding='utf-8') as f:
            charstext = f.read()
        # Extract all unique characters
        self.chars = sorted(list(set(charstext)))
        self.chars.extend([self.unk_token, self.eos_token])  # Add special tokens
        self.vocab_size = len(self.chars)

    def _create_mappings(self):
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.unk_id = self.stoi[self.unk_token]
        self.eos_id = self.stoi[self.eos_token]

    def encode(self, s):
        """Encodes a string into a list of token IDs."""
        # Split on EOS_TOKEN and add EOS_TOKEN at the end of each part
        parts = [
            [self.stoi.get(c, self.unk_id) for c in part] + [self.eos_id]
            for part in s.split(self.eos_token)
        ]
        # Flatten the list of lists
        return list(itertools.chain.from_iterable(parts))

    def decode(self, l):
        """Decodes a list of token IDs back into a string."""
        return ''.join([
            self.itos.get(i, self.unk_token) if i != self.eos_id
            else (self.eos_token if i == l[-1] else '')
            for i in l
        ])
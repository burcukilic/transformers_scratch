import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from blocks import *
from transformer_encoder import *
from transformer_decoder import *


class EncoderClassifier(nn.Module):
    def __init__(self, vocab_size: int = 10, embed_dim: int = 2, seq_len: int = 10, num_blocks: int = 2):
        
        super().__init__()

        self.encoder = TransformerEncoder(vocab_size=vocab_size, embed_dim=embed_dim, seq_len=seq_len, num_blocks=num_blocks)
        self.classifier = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x = self.encoder(x, mask=mask)
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)

        return x


def test_transformer_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test transformer encoder:
    def generate_batch():
        x = torch.randint(0, 100, (32, 20))  # Batch of 32 sequences of length 20
        # y is 1 if sorted else 0
        y = (torch.sort(x, dim=1).values == x).all(dim=1).long()
        mask = None  # No mask in this example
        return x, y, mask

    model = EncoderClassifier(vocab_size=100, embed_dim=64, seq_len=20, num_blocks=4).to(device)

    epochs = 20
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    lr = 1e-3

    for epoch in range(epochs):
        model.train()
        x, y, mask = generate_batch()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x, mask=mask)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

def test_transformer_decoder():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    vocab_size = 10
    embed_dim = 8
    seq_len = 5
    num_blocks = 4

    # Dummy dataset
    def generate_batch(batch_size=16, seq_len=5, vocab_size=5, pad_token=0, device='cpu'):
        input_seq = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        target_seq = (input_seq + 1) % vocab_size
        return input_seq, target_seq


    # Initialize decoder
    decoder = TransformerDecoder(vocab_size=vocab_size, embed_dim=embed_dim, seq_len=seq_len, num_blocks=num_blocks).to(device)

    # Training loop (very small overfit test)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)

    for step in range(2000):
        optimizer.zero_grad()
        input_seq, target_seq = generate_batch(batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size, device=device)
        loss = decoder.loss(input_seq, target_seq)
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    test_input = torch.tensor([[0, 1, 2, 3, 4]], device=device)
    pred = decoder(test_input).argmax(dim=-1)
    print("Input:     ", test_input.tolist())
    print("Predicted: ", pred.tolist())
    print("Target:    ", ((test_input + 1) % vocab_size).tolist())


def test_encoder_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_copy_task(batch_size=32, seq_len=5, vocab_size=10, bos_token=0, device='cpu'):
        # Input: random sequence
        src = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
        
        # Target = src (we want decoder to copy encoder input)
        tgt = src.clone()

        # Decoder input is BOS + src[:-1]
        bos = torch.full((batch_size, 1), bos_token, device=device)
        decoder_input = torch.cat([bos, tgt[:, :-1]], dim=1)

        return src, decoder_input, tgt
    
    def greedy_decode(model, src, bos_token=0, max_len=5):
        device = src.device
        batch_size = src.size(0)

        # Encode the input
        encoder_output = model.encoder(src)

        # Start decoder input with BOS token
        decoder_input = torch.full((batch_size, 1), bos_token, dtype=torch.long, device=device)

        for _ in range(max_len):
            # Predict logits for next token
            logits = model.decoder(decoder_input, encoder_output=encoder_output)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # take the last prediction

            # Append next token to decoder input
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

        return decoder_input[:, 1:]  # drop BOS

    vocab_size = 5
    seq_len = 5
    
    encoder = TransformerEncoder(vocab_size=vocab_size, embed_dim=8, seq_len=seq_len, num_blocks=2).to(device)
    decoder = TransformerDecoder(vocab_size=vocab_size, embed_dim=8, seq_len=seq_len, num_blocks=2).to(device)
    model = TransformerSeq2Seq(encoder, decoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for step in range(1000):
        src, tgt_in, tgt_out = generate_copy_task(batch_size=32, seq_len=seq_len, vocab_size=vocab_size, device=device)

        optimizer.zero_grad()
        loss = model.loss(src, tgt_in, tgt_out)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    src = torch.tensor([[1, 2, 3, 4, 1]], device=device)
    tgt_in = torch.tensor([[0, 1, 2, 3, 4]], device=device)
    tgt_out = src

    model.eval()
    with torch.no_grad():
        output = greedy_decode(model, src, bos_token=0, max_len=seq_len)

    print("Input:     ", src.tolist())
    print("Predicted: ", output.tolist())
    print("Target:    ", tgt_out.tolist())


def test_encoder_decoder_translation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torch.nn.utils.rnn import pad_sequence

    # Example toy dataset
    toy_dataset = [
        ("i am happy", "ich bin glücklich"),
        ("i am sad", "ich bin traurig"),
        ("i am angry", "ich bin wütend"),
        ("i am tired", "ich bin müde"),
        ("i am excited", "ich bin aufgeregt"),
        ("i am bored", "ich bin gelangweilt"),
        ("i am confused", "ich bin verwirrt"),
        ("i am surprised", "ich bin überrascht"),
        ("i am scared", "ich bin verängstigt"),
        ("i am relaxed", "ich bin entspannt"),
        ("i am proud", "ich bin stolz"),
        ("i am grateful", "ich bin dankbar"),
        ("i am curious", "ich bin neugierig"),
        ("i am hopeful", "ich bin hoffnungsvoll"),
        ("i am lonely", "ich bin einsam"),
        ("you are happy", "du bist glücklich"),
        ("you are sad", "du bist traurig"),
        ("you are angry", "du bist wütend"),
        ("you are tired", "du bist müde"),
        ("you are excited", "du bist aufgeregt"),
        ("you are bored", "du bist gelangweilt"),
        ("you are confused", "du bist verwirrt"),
        ("you are surprised", "du bist überrascht"),
        ("you are scared", "du bist verängstigt"),
        ("you are relaxed", "du bist entspannt"),
        ("you are proud", "du bist stolz"),
        ("you are grateful", "du bist dankbar"),
        ("you are curious", "du bist neugierig"),
        ("you are hopeful", "du bist hoffnungsvoll"),
        ("you are lonely", "du bist einsam"),
        ("they are happy", "sie sind glücklich"),
        ("they are sad", "sie sind traurig"),
        ("they are angry", "sie sind wütend"),  
        ("they are tired", "sie sind müde"),
        ("they are excited", "sie sind aufgeregt"),
        ("they are bored", "sie sind gelangweilt"),
        ("they are confused", "sie sind verwirrt"),
        ("they are surprised", "sie sind überrascht"),
        ("they are scared", "sie sind verängstigt"),
        ("they are relaxed", "sie sind entspannt"),
        ("they are proud", "sie sind stolz"),
        ("they are grateful", "sie sind dankbar"),
        ("they are curious", "sie sind neugierig"),
        ("they are hopeful", "sie sind hoffnungsvoll"),
        ("he is a student", "er ist ein Student"),
        ("she is a teacher", "sie ist eine Lehrerin"),
        ("we are friends", "wir sind Freunde"),
        ("you are family", "du bist Familie"),
        ("i am a doctor", "ich bin ein Arzt"),
        ("you are an engineer", "du bist ein Ingenieur"),
        ("they are artists", "sie sind Künstler"),
        ("we are musicians", "wir sind Musiker"),
        ("he is a scientist", "er ist ein Wissenschaftler"),
        ("she is a writer", "sie ist eine Schriftstellerin"),
        ("i am a chef", "ich bin ein Koch"),
        ("you are a nurse", "du bist eine Krankenschwester"),
        ("they are athletes", "sie sind Athleten"),
        ("we are dancers", "wir sind Tänzer"),
        ("he is a programmer", "er ist ein Programmierer"),
        ("she is a designer", "sie ist eine Designerin"),
        ("i am a photographer", "ich bin ein Fotograf"),
        ("you are a filmmaker", "du bist ein Filmemacher"),
        ("they are actors", "sie sind Schauspieler"),
        ("we are writers", "wir sind Schriftsteller"),
        ("he is a poet", "er ist ein Dichter"),
        ("she is a journalist", "sie ist eine Journalistin"),
        ("i am a student", "ich bin ein Student"),
        ("you are a teacher", "du bist ein Lehrer"),
        ("they are doctors", "sie sind Ärzte"),
        ("we are engineers", "wir sind Ingenieure"),
        ("he is an artist", "er ist ein Künstler"),
        ("she is a musician", "sie ist eine Musikerin"),
        ("i am a scientist", "ich bin ein Wissenschaftler"),
        ("you are a writer", "du bist ein Schriftsteller"),
        ("they are chefs", "sie sind Köche"),
        ("we are nurses", "wir sind Krankenschwestern"),
        ("he is an athlete", "er ist ein Athlet"),
        ("she is a dancer", "sie ist eine Tänzerin"),
        ("i am a programmer", "ich bin ein Programmierer"),
        ("you are a designer", "du bist ein Designer"),
        ("they are photographers", "sie sind Fotografen"),
        ("we are filmmakers", "wir sind Filmemacher"),
        ("he is an actor", "er ist ein Schauspieler"),
        ("she is a writer", "sie ist eine Schriftstellerin"),
        ("i am a poet", "ich bin ein Dichter"),
        ("you are a journalist", "du bist ein Journalist"),
        ("they are students", "sie sind Studenten"),
        ("we are teachers", "wir sind Lehrer"),
        ("he is a doctor", "er ist ein Arzt"),
        ("she is an engineer", "sie ist eine Ingenieurin"),
        ("i am an artist", "ich bin ein Künstler"),
        ("you are a musician", "du bist ein Musiker"),
        ("they are scientists", "sie sind Wissenschaftler"),
        ("we are writers", "wir sind Schriftsteller"),
        ("he is a chef", "er ist ein Koch"),
        ("she is a doctor", "sie ist eine Ärztin"),
        ("i am an engineer", "ich bin ein Ingenieur"),
        ("you are an artist", "du bist ein Künstler"),
        ("they are musicians", "sie sind Musiker"),
        ("we are scientists", "wir sind Wissenschaftler"),
        ("he is a writer", "er ist ein Schriftsteller"),
        ("she is a chef", "sie ist eine Köchin"),
        ("i am a nurse", "ich bin eine Krankenschwester"),
        ("you are an athlete", "du bist ein Athlet"),
        ("they are dancers", "sie sind Tänzer"),
        ("we are programmers", "wir sind Programmierer"),
        ("he is a designer", "er ist ein Designer"),
        ("she is a photographer", "sie ist eine Fotografin"),
        ("i am a filmmaker", "ich bin ein Filmemacher"),
        ("you are an actor", "du bist ein Schauspieler"),
        ("they are writers", "sie sind Schriftsteller"),
        ("we are poets", "wir sind Dichter"),
        ("he is a journalist", "er ist ein Journalist"),
        ("she is a student", "sie ist eine Studentin"),
        ("i am a teacher", "ich bin eine Lehrerin"),
        ("you are a doctor", "du bist ein Arzt"),
        ("they are engineers", "sie sind Ingenieure"),
        ("we are artists", "wir sind Künstler"),
        ("he is a musician", "er ist ein Musiker"),
        ("she is a scientist", "sie ist eine Wissenschaftlerin"),
        ("i am a turk", "ich bin ein Türke"),
        ("i am a german", "ich bin ein Deutscher"),
        ("you are a french", "du bist ein Franzose"),
        ("they are spanish", "sie sind Spanier"),
        ("we are italian", "wir sind Italiener"),
        ("he is british", "er ist Brite"),
        ("she is american", "sie ist Amerikanerin"),
        ("i am chinese", "ich bin Chinese"),
        ("you are japanese", "du bist Japaner"),
        ("they are korean", "sie sind Koreaner"),
        ("we are russian", "wir sind Russen"),
        ("he is indian", "er ist Inder"),
        ("she is brazilian", "sie ist Brasilianerin"),
        ("i am mexican", "ich bin Mexikaner"),
        ("you are canadian", "du bist Kanadier"),
        ("they are australian", "sie sind Australier"),
        ("you are german", "du bist Deutscher"),
        ("they are french", "sie sind Franzosen"),
        ("we are spanish", "wir sind Spanier"),
        ("he is italian", "er ist Italiener"),
        ("she is british", "sie ist Britin"),
        ("i am american", "ich bin Amerikaner"),
        ("you are chinese", "du bist Chinese"),
        ("they are japanese", "sie sind Japaner"),
        ("we are korean", "wir sind Koreaner"),
        ("he is russian", "er ist Russe"),
        ("she is indian", "sie ist Inderin"),
        ("i am brazilian", "ich bin Brasilianer"),
        ("you are mexican", "du bist Mexikaner"),
        ("they are canadian", "sie sind Kanadier"),
        ("we are australian", "wir sind Australier"),
        ("he is german", "er ist Deutscher"),
        ("she is french", "sie ist Französin"),
        ("i am spanish", "ich bin Spanier"),
        ("you are italian", "du bist Italiener"),
        ("they are british", "sie sind Briten"),
        ("we are american", "wir sind Amerikaner"),
        ("he is chinese", "er ist Chinese"),
        ("she is japanese", "sie ist Japanerin"),
        ("i am korean", "ich bin Koreaner"),
        ("you are russian", "du bist Russe"),
        ("they are indian", "sie sind Inder"),
        ("we are brazilian", "wir sind Brasilianer"),
        ("he is mexican", "er ist Mexikaner"),
        ("she is canadian", "sie ist Kanadierin"),
    ]

    from collections import Counter
    from itertools import chain

    def tokenize(text): return text.lower().split()

    def build_vocab(sentences, specials=["<pad>", "<bos>", "<eos>"]):
        counter = Counter(chain.from_iterable(tokenize(s) for s in sentences))
        vocab = {tok: i for i, tok in enumerate(specials)}
        vocab.update({tok: i+len(vocab) for i, tok in enumerate(counter)})
        itos = {i: tok for tok, i in vocab.items()}
        return vocab, itos

    # Build vocabs
    src_sentences = [src for src, _ in toy_dataset]
    tgt_sentences = [tgt for _, tgt in toy_dataset]
    vocab_src, itos_src = build_vocab(src_sentences)
    vocab_tgt, itos_tgt = build_vocab(tgt_sentences)

    pad_idx = vocab_tgt["<pad>"]
    bos_idx = vocab_tgt["<bos>"]
    eos_idx = vocab_tgt["<eos>"]


    # === Data Preprocessing ===
    def tensorize(text, vocab):
        tokens = tokenize(text)
        ids = [vocab["<bos>"]] + [vocab[t] for t in tokens] + [vocab["<eos>"]]
        return torch.tensor(ids, dtype=torch.long)

    def collate_toy_batch(batch):
        src_batch, tgt_batch = [], []
        for src, tgt in batch:
            src_tensor = tensorize(src, vocab_src)
            tgt_tensor = tensorize(tgt, vocab_tgt)
            src_batch.append(src_tensor)
            tgt_batch.append(tgt_tensor)
        src_batch = pad_sequence(src_batch, padding_value=vocab_src["<pad>"], batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=vocab_tgt["<pad>"], batch_first=True)
        return src_batch, tgt_batch

    
    def greedy_decode(model, src, max_len=20, bos_token=bos_idx):
        model.eval()
        with torch.no_grad():
            enc_output = model.encoder(src)
            batch_size = src.size(0)
            tgt_input = torch.full((batch_size, 1), bos_token, dtype=torch.long, device=src.device)

            for _ in range(max_len):
                logits = model.decoder(tgt_input, encoder_output=enc_output)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tgt_input = torch.cat([tgt_input, next_token], dim=1)
                if (next_token == eos_idx).all():
                    break
        return tgt_input


    from torch.utils.data import DataLoader

    loader = DataLoader(toy_dataset, batch_size=16, shuffle=True, collate_fn=collate_toy_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_dim = 32
    encoder = TransformerEncoder(vocab_size=len(vocab_src), embed_dim=embed_dim, seq_len=8, num_blocks=4).to(device)
    decoder = TransformerDecoder(vocab_size=len(vocab_tgt), embed_dim=embed_dim, seq_len=8, num_blocks=4).to(device)
    model = TransformerSeq2Seq(encoder, decoder).to(device)

    # load pretrained weights if available
    try:
        model.load_state_dict(torch.load("transformer_seq2seq_pretrained.pt", map_location=device))
        print("Loaded pretrained weights.")
        training = False
    except FileNotFoundError:
        print("No pretrained weights found, training from scratch.")
        training = True
    
    if training:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        print("Training...")
        model.train()
        for epoch in range(1000):
            for src, tgt in loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                loss = model.loss(src, tgt_input, tgt_output, pad_idx=pad_idx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
            # save model
            if (epoch + 1) % 100 == 0:
                torch.save(model.state_dict(), f"transformer_seq2seq_epoch_{epoch+1}.pt")

    # === Inference on 1 example ===
    src_text = "i am happy"
    src_tensor = tensorize(src_text, vocab_src).unsqueeze(0).to(device)
    output = greedy_decode(model, src_tensor)[0].tolist()
    translated = [itos_tgt[tok] for tok in output if tok not in {bos_idx, eos_idx, pad_idx}]
    print("\nInput:     ", src_text)
    print("Predicted: ", " ".join(translated))


test_encoder_decoder_translation()
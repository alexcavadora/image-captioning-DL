"""
Image Captioning Model: CNN Encoder + LSTM Decoder
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, pretrained=True, freeze=True):
        """
        CNN Encoder using pretrained ResNet

        Args:
            embed_size: Embedding dimension
            pretrained: Use pretrained weights
            freeze: Freeze CNN weights initially
        """
        super(EncoderCNN, self).__init__()

        # Load pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Remove final classification layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Projection to embedding space
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        # Freeze CNN parameters if specified
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, images):
        """
        Extract feature vectors from images

        Args:
            images: [batch_size, 3, 224, 224]

        Returns:
            features: [batch_size, embed_size]
        """
        with torch.no_grad():
            features = self.resnet(images)

        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)

        return features

    def unfreeze(self):
        """Unfreeze CNN parameters for fine-tuning"""
        for param in self.resnet.parameters():
            param.requires_grad = True
        print("[INFO] CNN Encoder unfrozen for fine-tuning")


class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.5):
        """
        LSTM Decoder for caption generation

        Args:
            embed_size: Word embedding dimension
            hidden_size: LSTM hidden state size
            vocab_size: Size of vocabulary
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(DecoderLSTM, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # LSTM core
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection to vocabulary
        self.linear = nn.Linear(hidden_size, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions, lengths):
        """
        Forward pass with teacher forcing

        Args:
            features: Image features [batch_size, embed_size]
            captions: Caption indices [batch_size, max_length]
            lengths: Caption lengths [batch_size]

        Returns:
            outputs: Word predictions [batch_size, max_length, vocab_size]
        """
        # Embed captions (exclude last token for input)
        embeddings = self.embedding(captions[:, :-1])

        # Concatenate image features as first input
        features = features.unsqueeze(1)  # [batch_size, 1, embed_size]
        embeddings = torch.cat([features, embeddings], dim=1)

        # Pack padded sequences for efficient LSTM processing
        # Subtract 1 from lengths because we concatenated features
        packed = pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=True
        )

        # LSTM forward pass
        hiddens, _ = self.lstm(packed)

        # Unpack sequence
        hiddens, _ = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True)

        # Project to vocabulary
        outputs = self.linear(self.dropout(hiddens))

        return outputs

    def sample(self, features, max_length=20):
        """
        Generate caption using greedy sampling (for inference)

        Args:
            features: Image features [1, embed_size]
            max_length: Maximum caption length

        Returns:
            caption: List of word indices
        """
        caption = []
        states = None

        inputs = features.unsqueeze(1)  # [1, 1, embed_size]

        for _ in range(max_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))

            # Get most likely word
            predicted = outputs.argmax(1)
            caption.append(predicted.item())

            # Stop if <end> token is generated
            if predicted.item() == 2:  # Assuming <end> is index 2
                break

            # Prepare next input
            inputs = self.embedding(predicted).unsqueeze(1)

        return caption

    def sample_beam_search(self, features, beam_width=3, max_length=20):
        """
        Generate caption using beam search

        Args:
            features: Image features [1, embed_size]
            beam_width: Beam width (k)
            max_length: Maximum caption length

        Returns:
            best_caption: List of word indices
        """
        k = beam_width
        # vocab_size = self.vocab_size

        # Start with initial input
        inputs = features.unsqueeze(1)  # [1, 1, embed_size]

        # Initialize beams: (score, caption, states)
        hiddens, states = self.lstm(inputs)
        outputs = self.linear(hiddens.squeeze(1))
        log_probs = torch.log_softmax(outputs, dim=1)

        # Get top k initial words
        top_log_probs, top_indices = log_probs.topk(k, dim=1)

        beams = []
        for i in range(k):
            caption = [top_indices[0, i].item()]
            score = top_log_probs[0, i].item()
            beams.append((score, caption, states))

        # Beam search
        for _ in range(max_length - 1):
            candidates = []

            for score, caption, prev_states in beams:
                # Stop if <end> token
                if caption[-1] == 2:
                    candidates.append((score, caption, prev_states))
                    continue

                # Get next word predictions
                word_idx = torch.tensor([caption[-1]]).to(features.device)
                inputs = self.embedding(word_idx).unsqueeze(1)
                hiddens, states = self.lstm(inputs, prev_states)
                outputs = self.linear(hiddens.squeeze(1))
                log_probs = torch.log_softmax(outputs, dim=1)

                # Get top k words
                top_log_probs, top_indices = log_probs.topk(k, dim=1)

                for i in range(k):
                    new_score = score + top_log_probs[0, i].item()
                    new_caption = caption + [top_indices[0, i].item()]
                    candidates.append((new_score, new_caption, states))

            # Select top k beams
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:k]

            # Check if all beams ended
            if all(caption[-1] == 2 for _, caption, _ in beams):
                break

        # Return best caption
        best_score, best_caption, _ = max(beams, key=lambda x: x[0])
        return best_caption


class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Complete Image Captioning Model

        Args:
            embed_size: Embedding dimension
            hidden_size: LSTM hidden size
            vocab_size: Vocabulary size
            num_layers: Number of LSTM layers
        """
        super(ImageCaptioningModel, self).__init__()

        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions, lengths):
        """Forward pass for training"""
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs

    def generate_caption(self, image, beam_search=False, beam_width=3, max_length=20):
        """Generate caption for a single image"""
        features = self.encoder(image)

        if beam_search:
            caption = self.decoder.sample_beam_search(features, beam_width, max_length)
        else:
            caption = self.decoder.sample(features, max_length)

        return caption

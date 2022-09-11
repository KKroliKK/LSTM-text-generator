import re
from typing import List
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from torch.optim import lr_scheduler
import torch
import argparse
from collections import Counter
from torch import nn
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


def parse_corpus(corpus):
    '''Разделяет корпус текста на отдельные анекдоты
    '''
    parsed = []
    buf = ''
    for line in corpus:
        match = re.fullmatch('\d+\n', line)
        if match:
            parsed.append(buf)
            buf = ''
        else:
            buf += line
    
    return parsed


stop_words = ['.', ',', '"', '!', "''", '%', '«', '»', '“', '”',
                 '?','(', ')', '-', '``', '..', '@', '#', "'", '—',
                 ':', ';', '_', '\\', '...', '\n', '*', '$', '=']


def nltk_tokenize(text):
    '''Токенизирует поданный анекдот
    '''
    tokens = word_tokenize(text, language="russian")
    tokens = [token.lower() for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if not re.match('\d', token)]
    return tokens



def tokenize_anecdots(path: str):
    '''Токенизирует корпус текста с анекдотами.
    '''
    with open(path, 'r') as fh:
        anecdotes = [line for line in fh]

    first_anec_idx = 76
    last_anec_idx = 31033
    anc = anecdotes[first_anec_idx : last_anec_idx]
    anc = [line for line in anc if line != '\n']

    parsed = parse_corpus(anc)

    tokenized = [nltk_tokenize(anek) for anek in parsed]

    return tokenized


class Dataset(torch.utils.data.Dataset):
    '''Представляет датасет.

        Args:
            path: Путь к корпусу текста.
            sequence_length: Длина последовательности слов, используемой при обучении сети.
        
        Attributes:
            sequence_length (int): Длина последовательности слов, используемой при обучении сети.
            words (list[str]): Лист, содержащий все слова корпуса текста подряд.
            uniq_words (list[str]): Лист, содержащий уникальные слова еорпуса текста.
            index_to_word (dict[int | str]): Словарь индекс:слово используется в методе генерации текста.
            word_to_index (dict[str | int]): Словарь слово:индекс для нахождения эмбеддинга слова.
            words_indexes (list[int]):  Индексы слов массива words, соответсвующие словарю word_to_index.
        '''
    def __init__(
        self,
        path: str,
        sequence_length: int=4,
    ):
        self.sequence_length = sequence_length
        self.words = self.load_words(path)
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self, path: str) -> List[str]:
        '''Загружает корпус текста по переданному пути.

        Args:
            path: Путь к файлу с корпусом текста.

        Returns:
            Лист, содержащий все слова корпуса текста.
        '''
        sentences = tokenize_anecdots(path)
        words = []
        for sentence in sentences:
            words += sentence
        return words

    def get_uniq_words(self):
        '''Возвращает лист, содержащий каждое уникальное слово корпуса текста.
        '''
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def get_random_word(self):
        '''Возвращает случайное слово из словаря корпуса текста.

        Необходим для генерации текста, если не была передана начальная последовательность.
        '''
        word = np.random.choice(self.uniq_words)
        return word

    def __len__(self):
        return len(self.words_indexes) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index : index + self.sequence_length]),
            torch.tensor(self.words_indexes[index + 1 : index + self.sequence_length + 1]),
        )


class Model(nn.Module):
    '''LSTM модель для генерации текста.

    Args:
        vocab_size: Размер словаря корпуса текста, на котором будет обучаться модель
        path: Путь к весам модели.
        
    Attributes:
        path (str): Путь к весам модели.
        lstm_size (int): Внутренний размер LSTM ячейки.
        embedding_dim (int): Размер вектора эмбеддингов слов.
        num_layers (int): Количество слоев LSTM
        embedding (torch.nn.Embedding): Модель эмбеддингов.
        lstm (torch.nn.LSTM): LSTM ячейки.
        fc (torch.nn.Linear): Классификатор.
    '''
    def __init__(self, vocab_size: int, path: str):
        super(Model, self).__init__()
        self.path = path
        self.lstm_size = 128
        self.embedding_dim = 64
        self.num_layers = 3

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, vocab_size)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)

        return logits, state

    def init_state(self, sequence_length: int):
        '''Возвращает случайные векторы для инициализации внутреннего состояния LSTM длиной
        sequence_length.
        '''
        return (torch.rand(self.num_layers, sequence_length, self.lstm_size),
                torch.rand(self.num_layers, sequence_length, self.lstm_size))

    def fit(
        self, 
        dataset: Dataset, 
        num_epochs: int=10, 
        batch_size: int=128, 
        sequence_length: int=4, 
        fine_tune: bool=False
        ):
        '''Метод обучения модели.

        Args:
            sequence_length: Длина последовательности слов, используемой при обучении сети.
            fine_tune: Если True, загружает веса уже обученной модели и продолжает обучение.
        '''
        self.to(device)
        if fine_tune:
            self.load_state_dict(torch.load(self.path, map_location=device))
        self.train()

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        state_h, state_c = self.init_state(sequence_length)
        state_h = state_h.to(device)
        state_c = state_c.to(device)

        smallest_loss = 1e9

        for epoch in range(num_epochs):
            batch_loss = 0
            for x, y in tqdm(dataloader):
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                y_pred, (state_h, state_c) = self(x, (state_h, state_c))
                loss = criterion(y_pred.transpose(1, 2), y)

                state_h = state_h.detach()
                state_c = state_c.detach()

                loss.backward()
                optimizer.step()

                batch_loss += loss.item()

                
            print(f'\nepoch: {epoch + 1}, loss: {batch_loss}\n\n')
            if batch_loss < smallest_loss:
                smallest_loss = batch_loss
                torch.save(self.state_dict(), self.path)

        self.load_state_dict(torch.load(self.path, map_location=device))

    def generate(
        self, 
        dataset: Dataset, 
        text: str=None, 
        next_words: int=50
        ) -> List[str]:
        '''Метод генерации текста.

        Args:
            dataset: Датасет, на котором была обучена модель.
            text: Начальная последовательность слов для генерации.
                Если значение None, модель сгенерирует текст, начиная
                со случайного слова.
            next_words: Сколько слов необходимо сгенерировать.

        Returns:
            Лист, содержащий начальную последовательность слов плюс
            сгенерированные слова.        
        '''
        self.load_state_dict(torch.load(self.path, map_location='cpu'))
        self.eval()

        if text is None:
            words = [dataset.get_random_word()]
        else:
            words = text.split(' ')

        state_h, state_c = self.init_state(1)
        y_pred = None

        for word in words:
            x = torch.tensor([[dataset.word_to_index[word]]])
            y_pred, (state_h, state_c) = self(x, (state_h, state_c))

        for i in range(next_words):
            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            words.append(dataset.index_to_word[word_index])

            x = torch.tensor([[dataset.word_to_index[words[-1]]]])
            y_pred, (state_h, state_c) = self(x, (state_h, state_c))

        return words


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='./data/anekdoty.txt')
    parser.add_argument('--model', type=str, default='model.pt')
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--fine-tune', type=bool, default=False)
    args = parser.parse_args()


    dataset = Dataset(args.input_dir)
    model = Model(len(dataset.uniq_words), args.model)

    model.fit(dataset, num_epochs=args.num_epochs, fine_tune=args.fine_tune)
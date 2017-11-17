# -*- encoding=utf-8 -*-
# Author: Sergio Burdisso (sergio.burdisso@gmail.com)
# Licence: MIT

from glob import glob

import io
import unicodedata
import numpy as np

ENCODING = "utf8"

# https://docs.python.org/2.4/lib/standard-encodings.html
CODECS = [
    "ascii", "utf_8", "cp1252", "big5", "big5hkscs", "cp037", "cp424",
    "cp500", "cp737", "cp775", "cp850", "cp852", "cp855", "cp856",
    "cp857", "cp860", "cp861", "cp862", "cp863", "cp864", "cp865",
    "cp866", "cp869", "cp874", "cp875", "cp932", "cp949", "cp950",
    "cp1006", "cp1026", "cp1140", "cp1250", "cp1251", "cp1253",
    "cp1254", "cp1255", "cp1256", "cp1257", "cp1258", "euc_jp",
    "euc_jis_2004", "euc_jisx0213", "euc_kr", "gb2312", "gbk", "gb18030",
    "hz", "iso2022_jp", "iso2022_jp_1", "iso2022_jp_2", "iso2022_jp_2004",
    "iso2022_jp_3", "iso2022_jp_ext", "iso2022_kr", "latin_1", "iso8859_2",
    "iso8859_3", "iso8859_4", "iso8859_5", "iso8859_6", "iso8859_7",
    "iso8859_8", "iso8859_9", "iso8859_10", "iso8859_13", "iso8859_14",
    "iso8859_15", "johab", "koi8_r", "koi8_u", "mac_cyrillic", "mac_greek",
    "mac_iceland", "mac_latin2", "mac_roman", "mac_turkish", "ptcp154",
    "shift_jis", "shift_jis_2004", "shift_jisx0213", "utf_16", "utf_16_be",
    "utf_16_le", "utf_7", "cp437"
]


class Dataset:
    ENCODER = {ord('\n'): 1}
    DECODER = {1: '\n'}
    UNK_CHAR = 0  # UNKOWN CHAR

    @staticmethod
    def __normalize_str__(str_text):
        return ''.join(
            c for c in unicodedata.normalize('NFKD', str_text)
            if unicodedata.category(c) != 'Mn'
        )

    @staticmethod
    def add_aplhabet_character(str_char):
        new_code = Dataset.get_alphabet_size() + 1
        Dataset.ENCODER[ord(str_char)] = new_code
        Dataset.DECODER[new_code] = str_char

    @staticmethod
    def get_alphabet_size():
        return (ord('~') - ord(' ') + 1) + len(Dataset.ENCODER) + 1  # (UNK_CHAR)

    @staticmethod
    def encode_char(str_char):
        char = ord(str_char)
        if 32 <= char <= 126:  # if ord(' ') <= char <= ord('~'):
            return char - 30
        else:
            return Dataset.ENCODER.get(char, Dataset.UNK_CHAR)

    @staticmethod
    def decode_char(enc_char):
        if enc_char in Dataset.DECODER:
            return Dataset.DECODER[enc_char]
        elif 2 <= enc_char <= 96:  # if ord(' ') <= char <= ord('~'):
            return chr(enc_char + 30)
        else:
            return ''

    @staticmethod
    def encode_str(str_text, normalize=True):
        if normalize:
            try:
                str_text = Dataset.__normalize_str__(str_text)
            except TypeError:
                str_text = Dataset.__normalize_str__(str_text.decode(ENCODING))
        return list(map(Dataset.encode_char, str_text))

    @staticmethod
    def load_from_files(path_pattern, str_eof_char=None, normalize=True):
        dataset = []
        for file_name in glob(path_pattern, recursive=True):
            the_file = io.open(file_name, "r", encoding=ENCODING)
            file_content = the_file.read()
            if str_eof_char:
                if ord(str_eof_char) not in Dataset.ENCODER:
                    Dataset.add_aplhabet_character(str_eof_char)
                file_content += str_eof_char
            dataset.extend(Dataset.encode_str(file_content, normalize))
            the_file.close()
            print("[dataset] file '%s' loaded." % file_name)

        print("[dataset] finished (%d characters loaded)." % len(dataset))
        return dataset

    @staticmethod
    def encode_files(path_pattern, target_codec="utf8", source_codec=None):
        guessed_codec = target_codec if not source_codec else source_codec

        for file_name in glob(path_pattern, recursive=True):
            print(
                "[dataset] encoding file '%s' from %s to %s."
                %
                (file_name, guessed_codec, target_codec)
            )
            decodeError = False
            try:
                the_file = io.open(file_name, "r", encoding=guessed_codec)
                file_content = the_file.read()
            except UnicodeDecodeError:
                decodeError = True

            if decodeError:
                print("[!] decode error: trying to guess codec...")
                for codec in CODECS:
                    try:
                        the_file = io.open(file_name, "r", encoding=codec)
                        file_content = the_file.read()
                        guessed_codec = codec
                        print("[!]    source codec changed to:", guessed_codec)
                        break
                    except UnicodeDecodeError:
                        pass

            the_file.close()
            the_file = io.open(file_name, "w", encoding=target_codec)
            the_file.write(file_content)
            the_file.close()

    @staticmethod
    def normalize_files(path_pattern):
        for file_name in glob(path_pattern, recursive=True):
            print("[dataset] normalizing file '%s'." % file_name)
            the_file = io.open(file_name, "r", encoding=ENCODING)
            file_content = the_file.read()
            the_file.close()

            the_file = io.open(file_name, "w", encoding=ENCODING)
            the_file.write(Dataset.__normalize_str__(file_content))
            the_file.close()

    @staticmethod
    def get_training_batches(n_epochs, batch_size, seq_length, dataset):
        dataset = np.array(dataset)
        n_dataset = len(dataset)
        n_batches = (n_dataset - 1) // (batch_size * seq_length)
        assert n_batches > 0, "Not enough data. Try a smaller batch_size?"
        rndata = n_batches * batch_size * seq_length
        X_reshaped = np.reshape(dataset[0:rndata], [batch_size, n_batches * seq_length])
        y_reshaped = np.reshape(dataset[1:rndata + 1], [batch_size, n_batches * seq_length])

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                X_batch = X_reshaped[:, batch * seq_length:(batch + 1) * seq_length]
                y_batch = y_reshaped[:, batch * seq_length:(batch + 1) * seq_length]
                X_batch = np.roll(X_batch, -epoch, axis=0)
                y_batch = np.roll(y_batch, -epoch, axis=0)
                yield epoch, X_batch, y_batch

    @staticmethod
    def peek_char_from_prob(probs, top_n=2):
        probs = np.squeeze(probs)
        probs[np.argsort(probs)[:-top_n]] = 0
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), 1, p=probs)[0]

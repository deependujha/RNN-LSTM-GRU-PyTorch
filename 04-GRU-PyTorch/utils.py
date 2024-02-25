"""_summary_
This file contains utility functions to load and process the data
"""

# data: https://download.pytorch.org/tutorial/data.zip
import io
import os
import unicodedata
import string
import glob

import random
import torch

# alphabet small + capital letters + " .,;'"
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)
category_lines = {}
all_categories = []


def unicode_to_ascii(s):
    """Turn a Unicode string to plain ASCII,
    thanks to https://stackoverflow.com/a/518232/2809427

    Args:
        s (string): unicode string

    Returns:
        string: ascii string
    """
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in ALL_LETTERS
    )


def load_data():
    """Build the category_lines dictionary, a list of names per language

    Returns:
        dict: category_lines
        list: all_categories
    """
    _category_lines = {}
    _all_categories = []

    def find_files(path):
        return glob.glob(path)

    # Read a file and split into lines
    def read_lines(filename):
        lines = io.open(filename, encoding="utf-8").read().strip().split("\n")
        return [unicode_to_ascii(line) for line in lines]

    for filename in find_files("../data/names/*.txt"):
        category = os.path.splitext(os.path.basename(filename))[0]
        _all_categories.append(category)

        lines = read_lines(filename)
        _category_lines[category] = lines

    return _category_lines, _all_categories


#!
#! ============================================================
# ? To represent a single letter, we use a “one-hot vector” of
# ? size `<1 x n_letters>`. A one-hot vector is filled with 0s
# ? except for a 1 at index of the current letter, e.g. "b" = <0 1 0 0 0 ...>.
#!
#! ============================================================
# ? To make a word we join a bunch of those into a
# ? 2D matrix <line_length x 1 x n_letters>.
#!
#! ============================================================
# ? That extra 1 dimension is because PyTorch assumes
# ? everything is in batches - we’re just using a batch size of 1 here.


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    """
    Convert a letter to its index in the alphabet
    """
    return ALL_LETTERS.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    """
    Convert a letter to its one-hot representation
    """
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


def line_to_tensor(line):
    """_summary_
    Turn a line into a <line_length x 1 x n_letters>,
    or an array of one-hot letter vectors
    """
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor


def sanitize_name_for_input(_name):
    """_summary_
    Sanitize the input name to be used as input to the model

    unicode -> ascii -> lower -> convert to tensor
    """
    _name = unicode_to_ascii(_name)
    _name = _name.lower()

    name_tensor = line_to_tensor(_name)
    return name_tensor


def sanitize_category_for_input(_category, _all_categories):
    """_summary_
    Sanitize the input category to be used for comparing with model prediction
    """
    _category_tensor = torch.tensor(
        [_all_categories.index(_category)], dtype=torch.long
    )
    return _category_tensor


def random_training_example(_category_lines, _all_categories):
    """_summary_
    Get a random category and random line from that category
    """

    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]

    category = random_choice(_all_categories)
    line = random_choice(_category_lines[category])
    category_tensor = torch.tensor([_all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


if __name__ == "__main__":
    print(ALL_LETTERS)
    print(unicode_to_ascii("Ślusàrski"))

    category_lines, all_categories = load_data()
    print(category_lines["Italian"][:5])

    print(letter_to_tensor("J"))  # [1, 57]
    print(line_to_tensor("Jones").size())  # [5, 1, 57]

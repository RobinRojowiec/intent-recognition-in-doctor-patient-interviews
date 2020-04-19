"""

IDE: PyCharm
Project: simulating-doctor-patient-interviews-using-neural-networks
Author: Robin
Filename: tables
Date: 26.04.2019

"""
import torch
from torchtext.data import Field


class Table:
    def __init__(self, column_headers, row_headers=None, class_field=None):
        self.data: [] = []
        self.column_headers = column_headers
        self.row_headers = row_headers
        self.class_field: Field = class_field

        if row_headers is not None:
            for i in range(len(self.row_headers)):
                self.add_row([0.0 for x in range(len(self.column_headers))])

    def header_exists(self, header_name):
        return header_name in self.column_headers

    def add_row(self, row: []):
        self.data.append(row)

    def get(self, x, y):
        return self.data[y][x]

    def get_by_header(self, column_header, row_header):
        x: int = self.get_x(column_header)
        y: int = self.get_y(row_header)
        return self.get(x, y), x, y

    def set(self, x, y, value):
        self.data[y][x] = value

    def get_x(self, header_name):
        return self.column_headers.index(header_name)

    def get_y(self, header_name):
        return self.row_headers.index(header_name)

    def get_column_sum(self, x, smoothing=0):
        return sum([row[x] + smoothing for row in self.data])

    def get_row_sum(self, y, smoothing=0):
        return sum([count + smoothing for count in self.data[y]])


class TransitionTable(Table):
    def __init__(self, states):
        Table.__init__(self, states, states)
        self.lambda_value = 1
        self.class_count = 63

    def batch_count_transitions(self, batch: []):
        for transition in batch:
            self.count_transition(transition[0], transition[1])

    def count_transition(self, state_a, state_b):
        value, x, y = self.get_by_header(state_a, state_b)
        self.set(x, y, value + 1)

    def get_probability(self, state_a, state_b):
        if (not self.header_exists(state_a)) or (not self.header_exists(state_b)):
            return 0.0
        else:
            transition_count, x, y = self.get_by_header(state_a, state_b)
            all_transitions = self.get_column_sum(x, self.lambda_value)
            probability: float = (transition_count + self.lambda_value) / (
                        all_transitions + (self.class_count * self.lambda_value))
            return probability

    def create_probability_matrix(self, device=torch.device('cpu')):
        tensor_stack = []
        num_labels = len(self.class_field.vocab.stoi)
        for state_b in range(num_labels):
            transitions_a_b = torch.zeros([num_labels])
            for state_a in range(num_labels):
                label_b, label_a = self.class_field.vocab.itos[state_b], self.class_field.vocab.itos[state_a]
                transitions_a_b[state_a] = self.get_probability(label_a, label_b)
            tensor_stack.append(transitions_a_b.to(device))
        return torch.stack(tensor_stack)

    def get_probability_tensor(self, states_a):
        transitions = []
        for state_a in states_a:
            transitions.append(self.tensor_cache[state_a])

        stacked = torch.stack(transitions)
        return stacked


if __name__ == '__main__':
    # Sequence: A, B, C
    tt = TransitionTable(["A", "B", "C"])

    test_transitions: [] = [["A", "B"], ["A", "B"], ["A", "C"], ["C", "A"]]
    tt.batch_count_transitions(test_transitions)

    print(tt.get_probability("A", "B"))
    print(tt.get_probability("C", "B"))

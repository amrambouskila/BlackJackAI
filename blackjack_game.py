# Standard library imports
import os
import pickle
import random
from collections import defaultdict, deque
from pathlib import Path

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Local application imports
import tkinter as tk
from tkinter import messagebox


def train_model(data_path: str = './data', epochs: int = 100):
    # Define the paths
    train_dir = f'{data_path}/train'
    valid_dir = f'{data_path}/valid'
    test_dir = f'{data_path}/test'

    # Load the CSV file
    dataset_csv = pd.read_csv(f'{data_path}/cards.csv')

    # Create ImageDataGenerator for loading and augmenting images
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    model_path = f'./models/{epochs}_model.h5'
    if Path(model_path).exists():
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(53, activation='softmax')  # 53 classes for 53 cards
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.load_weights(model_path)  # Load your trained model weights
        return model
    else:
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(53, activation='softmax')  # 53 classes for 53 cards
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=valid_generator
        )

        # Evaluate the model
        loss, accuracy = model.evaluate(test_generator)
        print(f'Test accuracy: {accuracy}')

        model.save(model_path)

    return model


class CardCounter:
    high_cards = ('ten', 'jack', 'queen', 'king', 'ace')
    low_cards = ('two', 'three', 'four', 'five', 'six')

    def __init__(self, model, n_decks: int = 8):
        self.model = model
        self.n_decks = n_decks
        self.total_cards = 52 * n_decks
        self._count = 0
        self._remaining_cards = {
            'total': self.total_cards,
            'high': int(round(self.total_cards * 5 / 13)),
            'low': int(round(self.total_cards * 5 / 13)),
            'neutral': int(round(self.total_cards * 3 / 13))
        }

        self.test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
            './data/test',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )

        self.card_counts = {label: 0 for label in self.test_generator.class_indices.keys()}

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, count: int):
        self._count = count

    @property
    def remaining_cards(self):
        return self._remaining_cards

    @remaining_cards.setter
    def remaining_cards(self, remaining_cards: int):
        self._remaining_cards = remaining_cards

    def reset_deck(self):
        self.total_cards = 52 * self.n_decks

        self.remaining_cards = {
            'total': self.total_cards,
            'high': int(round(self.total_cards * 5 / 13)),
            'low': int(round(self.total_cards * 5 / 13)),
            'neutral': int(round(self.total_cards * 3 / 13))
        }

        self.count = 0
        self.card_counts = {label: 0 for label in self.test_generator.class_indices.keys()}

    def classify_card(self, image):
        image = tf.image.resize(image, (224, 224))
        image = tf.expand_dims(image, 0)  # Add batch dimension
        predictions = self.model.predict(image)
        class_index = tf.argmax(predictions[0]).numpy()
        class_label = list(self.test_generator.class_indices.keys())[class_index]
        return class_label

    def count_cards(self, images):
        high_cards = 0
        low_cards = 0
        neutral_cards = 0
        card_labels = []
        for image in images:
            card_label = self.classify_card(image=image)
            rank = card_label.split()[0]
            suit = card_label.split()[-1]
            card_labels.append((rank, suit))
            self.card_counts[card_label] += 1

            if card_label.split()[0] in self.high_cards:
                self.count -= 1
                high_cards += 1
            elif card_label.split()[0] in self.low_cards:
                self.count += 1
                low_cards += 1
            else:
                neutral_cards += 1

        # Calculate remaining cards
        self.remaining_cards['total'] -= len(images)
        self.remaining_cards['high'] -= high_cards
        self.remaining_cards['low'] -= low_cards
        self.remaining_cards['neutral'] -= neutral_cards

        prob_high = self.remaining_cards['high'] / self.remaining_cards['total'] if self.remaining_cards[
                                                                                        'total'] > 0 else 0
        prob_low = self.remaining_cards['low'] / self.remaining_cards['total'] if self.remaining_cards[
                                                                                      'total'] > 0 else 0
        prob_neutral = self.remaining_cards['neutral'] / self.remaining_cards['total'] if self.remaining_cards[
                                                                                              'total'] > 0 else 0

        return prob_high, prob_low, prob_neutral, card_labels


class BlackjackGame:
    def __init__(self, root, bankroll: int, model_path: str):
        self.root = root
        self.root.title("Blackjack Game")

        self.suits = ('hearts', 'diamonds', 'clubs', 'spades')
        self.ranks = {
            'two': 2,
            'three': 3,
            'four': 4,
            'five': 5,
            'six': 6,
            'seven': 7,
            'eight': 8,
            'nine': 9,
            'ten': 10,
            'jack': 10,
            'queen': 10,
            'king': 10,
            'ace': 11
        }

        self.model_path = model_path

        # Load the trained model
        self.model = self.load_model()

        # Initialize game variables
        self.card_counter = CardCounter(model=self.model)
        self.deck = self.create_deck()
        self.player_hands = []
        self.dealer_hand = []
        self.card_images = []
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.bankroll = bankroll
        self.active_hand = 0  # Track active hand index (0 for player hand, 1+ for split hands)

        # Create GUI elements
        self.create_widgets()

    def load_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(53, activation='softmax')  # 53 classes for 53 cards
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.load_weights(self.model_path)  # Load your trained model weights
        return model

    def create_deck(self):
        deck = [(rank, suit) for rank in self.ranks.keys() for suit in self.suits for _ in
                range(self.card_counter.n_decks)]
        # random.shuffle(deck)
        return deck

    def create_widgets(self):
        self.player_label = tk.Label(self.root, text="Player's Hand")
        self.player_label.pack()

        self.frames_container = tk.Frame(self.root)
        self.frames_container.pack()

        self.player_frame = [tk.Frame(self.frames_container, padx=25) for i in range(4 * self.card_counter.n_decks) if
                             i != 0 else tk.Frame(self.frames_container, padx=25, bg='lightblue')]
        self.player_frame.pack(side=tk.LEFT)

        self.dealer_label = tk.Label(self.root, text="Dealer's Hand")
        self.dealer_label.pack()
        self.dealer_frame = tk.Frame(self.root)
        self.dealer_frame.pack()

        self.hit_button = tk.Button(self.root, text="Hit", command=self.hit)
        self.hit_button.pack(side=tk.LEFT)
        self.stand_button = tk.Button(self.root, text="Stand", command=self.stand)
        self.stand_button.pack(side=tk.LEFT)
        self.split_button = tk.Button(self.root, text="Split", command=self.split)  # Add split button
        self.split_button.pack(side=tk.LEFT)
        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_game)
        self.reset_button.pack(side=tk.LEFT)

        self.prob_label = tk.Label(self.root, text="Probabilities - High: 0.00, Low: 0.00, Neutral: 0.00")
        self.prob_label.pack()

        self.game_label = tk.Label(self.root, text="Game 1: Wins - 0, Losses - 0, Draws - 0 -- Cards Left = 416")
        self.game_label.pack()

        self.wallet_label = tk.Label(self.root, text=f"Bankroll: ${self.bankroll}")
        self.wallet_label.pack()

    def start_game(self):
        self.deal_card_to_player()
        self.deal_card_to_dealer()
        self.deal_card_to_player()
        self.deal_card_to_dealer()

        player_value = self.calculate_hand_value(self.player_hand)
        dealer_value = self.calculate_hand_value(self.dealer_hand)
        if player_value == 21:
            self.wins += 1
            self.update_game_label(150)
            messagebox.showinfo("Blackjack", "Blackjack! Player Wins!")
            self.reset_hand()
        elif dealer_value == 21:
            self.losses += 1
            self.update_game_label(-100)
            messagebox.showinfo("Blackjack", "Dealer wins!")
            self.reset_hand()

        # Enable the split button if the player has two cards of the same rank
        self.update_split_button_state()

    def update_split_button_state(self):
        can_split = any(len(hand) == 2 and hand[0][0] == hand[1][0] for hand in [self.player_hand] + self.split_hands)
        self.split_button.config(state=tk.NORMAL if can_split else tk.DISABLED)

    def split(self):
        # Check if the active hand can be split
        active_hand = self.player_hands[self.active_hand]
        if len(active_hand) == 2 and active_hand[0][0] == active_hand[1][0]:
            new_hand = [self.player_hands[self.active_hand].pop()]

        if self.active_hand == 0:
            if len(self.player_hand) == 2 and self.player_hand[0][0] == self.player_hand[1][0]:
                self.split_hands.append([self.player_hand.pop()])
                self.deal_card_to_player()
                self.deal_card_to_split_hand(self.split_hands[-1])
        elif self.active_hand <= len(self.split_hands[self.active_hand - 1]):
            if len(self.split_hands[self.active_hand - 1]) == 2 and self.split_hands[self.active_hand - 1][0][0] == \
                    self.split_hands[self.active_hand - 1][1][0]:
                self.split_hands.append([self.split_hands[self.active_hand - 1].pop()])
                self.deal_card_to_split_hand(self.split_hands[-1])
                self.deal_card_to_split_hand(self.split_hands[self.active_hand - 1])

        self.update_split_button_state()
        for split_frame in self.split_frames[:len(self.split_hands)]:
            split_frame.pack(side=tk.LEFT, padx=10)

        self.update_hand_display()

    def deal_card_to_split_hand(self, hand):
        card = self.deck.pop()
        self.card_counter.total_cards -= 1
        if self.card_counter.total_cards == 0:
            self.deck = self.create_deck()
            self.card_counter.reset_deck()

        hand.append(card)
        img_path = f'./data/test/{card[0]} of {card[1]}/1.jpg'

        if not os.path.exists(img_path):
            print(f"Image path does not exist: {img_path}")
            return

        images = [tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)]
        prob_high, prob_low, prob_neutral, predicted_cards = self.card_counter.count_cards(images=images)

        try:
            img = Image.open(img_path)
            img = img.resize((100, 150), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.card_images.append(photo)
            frame = tk.Frame(self.split_frames[len(self.split_hands) - 1])
            frame.pack(side=tk.LEFT)
            label = tk.Label(frame, image=photo)
            label.image = photo
            label.pack()
            tk.Label(frame, text=f'{card[0]} of {card[1]}').pack()
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")

        self.update_probabilities_label(prob_high, prob_low, prob_neutral)

    def update_probabilities_label(self, prob_high, prob_low, prob_neutral):
        self.prob_label.config(
            text=f'Probabilities - High: {prob_high:.2f}, Low: {prob_low:.2f}, Neutral: {prob_neutral:.2f}')

    def update_game_label(self, rewards: int):
        self.game_label.config(
            text=f'Game {self.wins + self.losses + self.draws}: Wins - {self.wins}, Losses - {self.losses}, Draws - {self.draws} -- Cards Left = {self.card_counter.remaining_cards["total"]}')
        self.bankroll += rewards
        self.wallet_label.config(text=f"Bankroll: ${self.bankroll}")

    def highlight_active_hand(self):
        # Reset all frames' background color
        self.player_frame.config(bg='lightblue' if self.active_hand == 0 else 'SystemButtonFace')
        for i, frame in enumerate(self.split_frames):
            if i < len(self.split_hands):
                frame.config(bg='lightblue' if self.active_hand == i + 1 else 'SystemButtonFace')

    def update_hand_display(self):
        for widget in self.player_frame.winfo_children():
            widget.destroy()
        for split_frame in self.split_frames:
            for widget in split_frame.winfo_children():
                widget.destroy()

        for card in self.player_hand:
            img_path = f'./data/test/{card[0]} of {card[1]}/1.jpg'
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    img = img.resize((100, 150), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.card_images.append(photo)
                    frame = tk.Frame(self.player_frame)
                    frame.pack(side=tk.LEFT)
                    label = tk.Label(frame, image=photo)
                    label.image = photo
                    label.pack()
                    tk.Label(frame, text=f'{card[0]} of {card[1]}').pack()
                except Exception as e:
                    print(f"Failed to load image {img_path}: {e}")

        for i, hand in enumerate(self.split_hands):
            for card in hand:
                img_path = f'./data/test/{card[0]} of {card[1]}/1.jpg'
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        img = img.resize((100, 150), Image.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        self.card_images.append(photo)
                        frame = tk.Frame(self.split_frames[i])
                        frame.pack(side=tk.LEFT)
                        label = tk.Label(frame, image=photo)
                        label.image = photo
                        label.pack()
                        tk.Label(frame, text=f'{card[0]} of {card[1]}').pack()
                    except Exception as e:
                        print(f"Failed to load image {img_path}: {e}")

        self.highlight_active_hand()

    def deal_card_to_player(self):
        card = self.deck.pop()
        self.card_counter.total_cards -= 1
        if self.card_counter.total_cards == 0:
            self.deck = self.create_deck()
            self.card_counter.reset_deck()

        self.player_hand.append(card)
        img_path = f'./data/test/{card[0]} of {card[1]}/1.jpg'

        if not os.path.exists(img_path):
            print(f"Image path does not exist: {img_path}")
            return

        images = [tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)]
        prob_high, prob_low, prob_neutral, predicted_cards = self.card_counter.count_cards(images=images)

        try:
            img = Image.open(img_path)
            img = img.resize((100, 150), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.card_images.append(photo)
            frame = tk.Frame(self.player_frame)
            frame.pack(side=tk.LEFT)
            label = tk.Label(frame, image=photo)
            label.image = photo
            label.pack()
            tk.Label(frame, text=f'{card[0]} of {card[1]}').pack()
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")

        self.update_probabilities_label(prob_high, prob_low, prob_neutral)

    def deal_card_to_dealer(self):
        card = self.deck.pop()
        self.card_counter.total_cards -= 1
        if self.card_counter.total_cards == 0:
            self.deck = self.create_deck()
            self.card_counter.reset_deck()

        self.dealer_hand.append(card)
        img_path = f'./data/test/{card[0]} of {card[1]}/1.jpg'

        if not os.path.exists(img_path):
            print(f"Image path does not exist: {img_path}")
            return

        images = [tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)]
        prob_high, prob_low, prob_neutral, predicted_cards = self.card_counter.count_cards(images=images)

        try:
            img = Image.open(img_path)
            img = img.resize((100, 150), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.card_images.append(photo)
            frame = tk.Frame(self.dealer_frame)
            frame.pack(side=tk.LEFT)
            label = tk.Label(frame, image=photo)
            label.image = photo
            label.pack()
            tk.Label(frame, text=f'{card[0]} of {card[1]}').pack()
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")

        self.update_probabilities_label(prob_high, prob_low, prob_neutral)

    def hit(self):
        if self.active_hand == 0:
            # Player's main hand
            self.deal_card_to_player()
            player_value = self.calculate_hand_value(self.player_hand)
            if player_value >= 21:
                self.active_hand += 1
                self.highlight_active_hand()
        else:
            # One of the split hands
            self.deal_card_to_split_hand(self.split_hands[self.active_hand - 1])
            hand_value = self.calculate_hand_value(self.split_hands[self.active_hand - 1])
            if hand_value >= 21:
                self.active_hand += 1
                self.highlight_active_hand()
                if self.active_hand > len(self.split_hands):
                    self.stand()
            else:
                self.highlight_active_hand()

    def stand(self):
        if self.active_hand <= len(self.split_hands):
            self.active_hand += 1
            self.highlight_active_hand()
            if self.active_hand <= len(self.split_hands):
                return

        # If no more hands to play, process the dealer's turn
        while self.calculate_hand_value(self.dealer_hand) < 17:
            self.deal_card_to_dealer()

        dealer_value = self.calculate_hand_value(self.dealer_hand)
        player_value = self.calculate_hand_value(self.player_hand)
        split_values = [self.calculate_hand_value(hand) for hand in self.split_hands]

        # Compare dealer's hand with player's hands
        message = self.compare_hands(player_value, dealer_value, "Player")
        for idx, split_value in enumerate(split_values):
            message += self.compare_hands(split_value, dealer_value, f"Split {idx + 1}")

        messagebox.showinfo("Blackjack", message)
        self.reset_hand()

    def compare_hands(self, player_value, dealer_value, hand_name):
        player_blackjack_win = player_value == 21
        player_regular_win = (player_value < 21 and (dealer_value > 21 or player_value > dealer_value))
        draw = player_value < 21 and player_value == dealer_value
        dealer_wins = dealer_value <= 21 and player_value < dealer_value

        message = ""
        if player_blackjack_win:
            self.wins += 1
            self.update_game_label(150)
            message += f"{hand_name} Blackjack! Player wins!\n"
        elif player_regular_win:
            self.wins += 1
            self.update_game_label(100)
            message += f"{hand_name} Value: {player_value} > Dealer Value: {dealer_value}\n"
        elif draw:
            self.draws += 1
            message += f"{hand_name} Value: {player_value} == Dealer Value: {dealer_value}\n"
        elif dealer_wins:
            self.losses += 1
            self.update_game_label(-100)
            message += f"Dealer Value: {dealer_value} > {hand_name} Value: {player_value}\n"

        return message

    def reset_hand(self):
        for widget in self.player_frame.winfo_children():
            widget.destroy()

        for split_frame in self.split_frames:
            for widget in split_frame.winfo_children():
                widget.destroy()

        for widget in self.dealer_frame.winfo_children():
            widget.destroy()

        self.player_hand = []
        self.dealer_hand = []
        self.split_hands = []
        self.active_hand = 0
        for split_frame in self.split_frames:
            split_frame.pack_forget()
        self.start_game()
        self.enable_buttons()

    def calculate_hand_value(self, hand):
        value = 0
        ace_count = 0
        for card in hand:
            rank = card[0]
            value += self.ranks[rank]

            if rank == 'ace':
                ace_count += 1

        while value > 21 and ace_count:
            value -= 10
            ace_count -= 1

        return value

    def disable_buttons(self):
        self.hit_button.config(state=tk.DISABLED)
        self.stand_button.config(state=tk.DISABLED)

    def enable_buttons(self):
        self.hit_button.config(state=tk.NORMAL)
        self.stand_button.config(state=tk.NORMAL)

    def reset_game(self):
        self.deck = self.create_deck()
        self.card_counter.reset_deck()


if __name__ == '__main__':
    model = train_model(epochs=5)
    image_paths = [r'.\data\test\ace of clubs\1.jpg', r'.\data\test\nine of clubs\1.jpg',
                   r'.\data\test\six of clubs\1.jpg', r'.\data\test\king of hearts\1.jpg']

    images = [tf.io.read_file(image_path) for image_path in image_paths]
    images = [tf.image.decode_jpeg(image, channels=3) for image in images]
    card_counter = CardCounter(model=model)

    prob_high, prob_low, prob_neutral, card_labels = card_counter.count_cards(images=images)
    print(f'Predicted Cards: {card_labels}')
    print(f'Probability of next card being +1 count: {prob_low:.2f}')
    print(f'Probability of next card being -1 count: {prob_high:.2f}')
    print(f'Probability of next card being 0 count: {prob_neutral:.2f}')

    # Create the main window
    root = tk.Tk()
    game = BlackjackGame(root, 18000, './models/5_model.h5')
    game.start_game()
    root.mainloop()
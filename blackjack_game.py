# Standard library imports
import os
import pickle
import random
from collections import defaultdict, deque
from pathlib import Path

# Third-party library imports
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Local application imports
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox, ttk


matplotlib.use('Qt5Agg', force=True)
sns.set()


# Custom ReLU with B-spline parameters
class BSplineReLU(Layer):
    def __init__(self, **kwargs):
        super(BSplineReLU, self).__init__(**kwargs)
        self.control_points = self.add_weight(name='control_points', shape=(4,), initializer='ones', trainable=True)

    def call(self, inputs):
        # Example logic to incorporate B-spline control points into ReLU (simplified for illustration)
        relu_output = tf.nn.relu(inputs)
        spline_output = relu_output * self.control_points[0] + (1 - relu_output) * self.control_points[1]
        return spline_output


# Custom KAN layer
class KANConv2D(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(KANConv2D, self).__init__(**kwargs)
        self.conv = Conv2D(filters, kernel_size, **kwargs)
        self.b_spline_relu = BSplineReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        return self.b_spline_relu(x)


# KAN model
def KANCNN(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = KANConv2D(32, (3, 3))(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = KANConv2D(64, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)
    x = KANConv2D(128, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = BSplineReLU()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model


# Function to update the plot
def update_plot(fig, epoch, train_losses, val_losses, lrs, weights_stats):
    plt.clf()

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.plot(range(1, len(lrs) + 1), lrs, label='Learning Rate')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Learning Rate')
    plt.legend()

    # Plot weight statistics
    plt.subplot(1, 2, 2)
    if len(weights_stats) > 0:
        for layer_idx in range(len(weights_stats[0])):
            means = [stat[layer_idx]['mean'] for stat in weights_stats]
            stds = [stat[layer_idx]['std'] for stat in weights_stats]
            plt.plot(range(1, len(weights_stats) + 1), means, label=f'Weights Layer {layer_idx} Mean')
            plt.fill_between(range(1, len(weights_stats) + 1), np.array(means) - np.array(stds),
                             np.array(means) + np.array(stds), alpha=0.2)

    plt.title('Weights Mean and Std Dev Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    plt.suptitle(f'Epoch {epoch + 1}')
    plt.pause(0.1)
    fig.canvas.draw()


# Custom training loop
def train_model(data_path: str = './data', epochs: int = 100, use_kan: bool = True):
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
    input_shape = (224, 224, 3)
    num_classes = 53  # 53 classes for 53 cards

    if Path(model_path).exists():
        model = KANCNN(input_shape, num_classes) if use_kan else Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.load_weights(model_path)  # Load your trained model weights
        return model
    else:
        model = KANCNN(input_shape, num_classes) if use_kan else Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Custom training loop to update B-spline parameters and weights
        train_losses = []
        val_losses = []
        lrs = []
        control_points_values = []
        weights_stats = []

        fig = plt.figure(figsize=(16, 6))
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            history = model.fit(train_generator, validation_data=valid_generator, epochs=1)

            train_loss = history.history['loss'][0]
            val_loss = history.history['val_loss'][0]
            lr = model.optimizer.learning_rate.numpy()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            lrs.append(lr)

            # Collect control points and weight statistics
            control_points = []
            layer_weights_stats = []
            for layer in model.layers:
                if isinstance(layer, BSplineReLU):
                    control_points.append(layer.control_points.numpy())
                if isinstance(layer, KANConv2D):
                    weights = layer.conv.get_weights()[0]
                    mean = np.mean(weights)
                    std = np.std(weights)
                    layer_weights_stats.append({'mean': mean, 'std': std})

            control_points_values.append(control_points)
            weights_stats.append(layer_weights_stats)

            update_plot(fig, epoch, train_losses, val_losses, lrs, weights_stats)

            # Update B-spline parameters and control points
            for layer in model.layers:
                if isinstance(layer, KANConv2D) or isinstance(layer, BSplineReLU):
                    with tf.GradientTape() as tape:
                        tape.watch([tf.convert_to_tensor(w) for w in layer.trainable_weights])
                        predictions = model(train_generator[0][0])
                        loss = tf.keras.losses.categorical_crossentropy(train_generator[0][1], predictions)
                    grads = tape.gradient(loss, layer.trainable_weights)
                    for weight, grad in zip(layer.trainable_weights, grads):
                        if isinstance(weight, tf.Variable):
                            weight.assign_sub(grad * 0.01)  # Update step size

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

        prob_high = self.remaining_cards['high'] / self.remaining_cards['total'] if self.remaining_cards['total'] > 0 else 0
        prob_low = self.remaining_cards['low'] / self.remaining_cards['total'] if self.remaining_cards['total'] > 0 else 0
        prob_neutral = self.remaining_cards['neutral'] / self.remaining_cards['total'] if self.remaining_cards['total'] > 0 else 0

        return prob_high, prob_low, prob_neutral, card_labels


class BlackjackGame:
    def __init__(self, root, bankroll: int, model_path: str, cost: int = 100):
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

        # Load the trained model
        self.model_path = model_path
        self.model = self.load_model()

        # Initialize the card counter
        self.card_counter = CardCounter(model=self.model)
        self.deck = self.create_deck()

        # Initialize game variables
        self.starting_bankroll = bankroll
        self._bankroll = bankroll
        self._cost = cost
        self._blackjack_wins = 0
        self._regular_wins = 0
        self._losses = 0
        self._draws = 0
        self._hits = 0
        self._splits = 0
        self._stands = 0
        self._hand_count = 0
        self._active_hand = 0
        self._dealer_hand = []
        self._card_images = []
        self._player_hands = [[]]
        self._probabilities = {'High': [5 / 13], 'Low': [5 / 13], 'Neutral': [3 / 13]}
        self._available_rewards = {'Blackjack': int(cost * 1.5), 'Regular': cost, 'Loss': -cost}
        self.game_log = []

        # Create a main container frame to hold player and dealer frames
        self.main_container = tk.Frame(self.root)
        self.main_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.player_label = tk.Label(self.main_container, text="Player's Hand")
        self.player_label.pack()

        self.frames_container = tk.Frame(self.main_container)
        self.frames_container.pack()

        self._player_frames = [
            tk.Frame(self.frames_container, padx=25) if i != 0 else tk.Frame(self.frames_container, padx=25,
                                                                             bg='lightblue') for i in
            range(4 * self.card_counter.n_decks)]
        for frame in self._player_frames:
            frame.pack(side=tk.LEFT)

        self.dealer_label = tk.Label(self.main_container, text="Dealer's Hand")
        self.dealer_label.pack()
        self.dealer_frame = tk.Frame(self.main_container)
        self.dealer_frame.pack()

        self.hit_button = tk.Button(self.main_container, text="Hit", command=self.hit)
        self.hit_button.pack(side=tk.LEFT)
        self.stand_button = tk.Button(self.main_container, text="Stand", command=self.stand)
        self.stand_button.pack(side=tk.LEFT)
        self.split_button = tk.Button(self.main_container, text="Split", command=self.split)  # Add split button
        self.split_button.pack(side=tk.LEFT)
        self.reset_button = tk.Button(self.main_container, text="Reset", command=self.reset_game)
        self.reset_button.pack(side=tk.LEFT)

        self.prob_label = tk.Label(self.main_container, text=f"Probabilities - High: {self._probabilities['High'][0]}, Low: {self._probabilities['High'][0]}, Neutral: {self._probabilities['High'][0]}")
        self.prob_label.pack()

        self.game_label = tk.Label(self.main_container, text=f'Game {self.regular_wins + self.blackjack_wins + self.losses + self.draws}: Blackjack Wins - {self.blackjack_wins}, Regular Wins - {self.regular_wins}, Losses - {self.losses}, Draws - {self.draws} -- Cards Left = {self.card_counter.remaining_cards["total"]}')
        self.game_label.pack()

        self.wallet_label = tk.Label(self.main_container, text=f"Bankroll: ${self.bankroll}")
        self.wallet_label.pack()

        # Create the frame for the plots and logs
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Plot frame should contain both the canvas and the log_frame side by side
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a scrollable text widget for logs
        self.log_frame = tk.Frame(self.plot_frame)
        self.log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(self.log_frame, wrap=tk.WORD, state=tk.NORMAL)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_scrollbar = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=self.log_scrollbar.set)

        # Initialize data for plots
        self._returns = []
        self._average_returns = []

        # Initialize the animation
        self.ani = FuncAnimation(self.fig, self.update_plots, interval=1000)

    @property
    def bankroll(self):
        return self._bankroll

    @bankroll.setter
    def bankroll(self, bankroll):
        self._bankroll = bankroll

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, cost):
        self._cost = cost

    @property
    def blackjack_wins(self):
        return self._blackjack_wins

    @blackjack_wins.setter
    def blackjack_wins(self, blackjack_wins):
        self._blackjack_wins = blackjack_wins

    @property
    def regular_wins(self):
        return self._regular_wins

    @regular_wins.setter
    def regular_wins(self, regular_wins):
        self._regular_wins = regular_wins

    @property
    def losses(self):
        return self._losses

    @losses.setter
    def losses(self, losses):
        self._losses = losses

    @property
    def draws(self):
        return self._draws

    @draws.setter
    def draws(self, draws):
        self._draws = draws

    @property
    def hits(self):
        return self._hits

    @hits.setter
    def hits(self, hits):
        self._hits = hits

    @property
    def splits(self):
        return self._splits

    @splits.setter
    def splits(self, splits):
        self._splits = splits

    @property
    def stands(self):
        return self._stands

    @stands.setter
    def stands(self, stands):
        self._stands = stands

    @property
    def hand_count(self):
        return self._hand_count

    @hand_count.setter
    def hand_count(self, hand_count):
        self._hand_count = hand_count

    @property
    def active_hand(self):
        return self._active_hand

    @active_hand.setter
    def active_hand(self, active_hand):
        self._active_hand = active_hand

    @property
    def dealer_hand(self):
        return self._dealer_hand

    @dealer_hand.setter
    def dealer_hand(self, dealer_hand):
        self._dealer_hand = dealer_hand

    @property
    def card_images(self):
        return self._card_images

    @card_images.setter
    def card_images(self, card_images):
        self._card_images = card_images

    @property
    def player_hands(self):
        return self._player_hands

    @player_hands.setter
    def player_hands(self, player_hands):
        self._player_hands = player_hands

    @property
    def player_frames(self):
        return self._player_frames

    @player_frames.setter
    def player_frames(self, player_frames):
        self._player_frames = player_frames

    @property
    def probabilities(self):
        return self._probabilities

    @probabilities.setter
    def probabilities(self, probabilities):
        self._probabilities = probabilities

    @property
    def available_rewards(self):
        return self._available_rewards

    @available_rewards.setter
    def available_rewards(self, available_rewards):
        self._available_rewards = available_rewards

    @property
    def returns(self):
        return self._returns

    @returns.setter
    def returns(self, returns):
        self._returns = returns

    @property
    def average_returns(self):
        return self._average_returns

    @average_returns.setter
    def average_returns(self, average_returns):
        self._average_returns = average_returns

    def log_prefix(self, idx: int):
        return f'Hand {self.hand_count}: Active Hand {idx + 1} -'

    def load_model(self, use_kan: bool = True):
        input_shape = (224, 224, 3)
        num_classes = 53  # 53 classes for 53 cards

        model = KANCNN(input_shape, num_classes) if use_kan else Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
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
        random.shuffle(deck)
        return deck

    def display_hand(self, card, is_dealer):
        img_path = f'./data/test/{card[0]} of {card[1]}/1.jpg'

        if not os.path.exists(img_path):
            print(f"Image path does not exist: {img_path}")
            return

        images = [tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)]
        prob_high, prob_low, prob_neutral, predicted_cards = self.card_counter.count_cards(images=images)
        self.probabilities['High'].append(prob_high)
        self.probabilities['Low'].append(prob_low)
        self.probabilities['Neutral'].append(prob_neutral)

        try:
            img = Image.open(img_path)
            img = img.resize((100, 150), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.card_images.append(photo)
            frame = tk.Frame(self.dealer_frame if is_dealer else self.player_frames[self.active_hand])
            frame.pack(side=tk.LEFT)
            label = tk.Label(frame, image=photo)
            label.image = photo
            label.pack()
            frame_text = f'{card[0]} of {card[1]}'
            if is_dealer:
                frame_text = f'Actual: {frame_text}\nPredicted: {predicted_cards[0][0]} of {predicted_cards[0][1]}'

            tk.Label(frame, text=frame_text).pack()
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")

        self.update_probabilities_label()

    def deal_card(self, hand: list, is_dealer: bool = False):
        card = self.deck.pop()
        self.card_counter.total_cards -= 1
        if self.card_counter.total_cards == 0:
            self.deck = self.create_deck()
            self.card_counter.reset_deck()
            self.probabilities = {'High': [5 / 13], 'Low': [5 / 13], 'Neutral': [3 / 13]}
            self.card_images = []

        hand.append(card)
        self.display_hand(card=card, is_dealer=is_dealer)

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

    def update_probabilities_label(self):
        self.prob_label.config(text=f'Probabilities - High: {self.probabilities["High"][-1]:.2f}, Low: {self.probabilities["Low"][-1]:.2f}, Neutral: {self.probabilities["Neutral"][-1]:.2f}')

    def update_game_label(self, rewards: int):
        self.game_label.config(text=f'Game {self.regular_wins + self.blackjack_wins + self.losses + self.draws}: Blackjack Wins - {self.blackjack_wins}, Regular Wins - {self.regular_wins}, Losses - {self.losses}, Draws - {self.draws} -- Cards Left = {self.card_counter.remaining_cards["total"]}')
        self.bankroll += rewards
        self.wallet_label.config(text=f"Bankroll: ${self.bankroll}")

    def update_plots(self, *args):
        win_data = {
            "regular_wins": self.regular_wins,
            "blackjack_wins": self.blackjack_wins,
            "losses": self.losses,
            "draws": self.draws
        }

        prob_data = {
            "high": self.probabilities["High"][-1],
            "low": self.probabilities["Low"][-1],
            "neutral": self.probabilities["Neutral"][-1]
        }

        action_data = {
            "hits": self.hits,
            "stands": self.stands,
            "splits": self.splits
        }

        # Clear previous plots
        for ax in self.axs.flat:
            ax.clear()

        # Plot 1: Wins, Losses, Draws
        self.axs[0, 0].bar(win_data.keys(), win_data.values())
        self.axs[0, 0].set_title("Wins, Losses, Draws")
        self.axs[0, 0].set_ylabel("Count")
        self.axs[0, 0].tick_params(axis='x', rotation=45)

        # Plot 2: Probabilities
        self.axs[0, 1].bar(prob_data.keys(), prob_data.values())
        self.axs[0, 1].set_title("Probabilities")
        self.axs[0, 1].set_ylabel("Probability")
        self.axs[0, 1].tick_params(axis='x', rotation=45)

        # Plot 3: Average Returns
        if self.hand_count > 0:
            self.axs[1, 0].plot(range(1, len(self.average_returns) + 1), self.average_returns)
        self.axs[1, 0].set_title("Average Returns per Hand")
        self.axs[1, 0].set_ylabel("Average Return")
        self.axs[1, 0].set_xlabel("Number of Hands")
        self.axs[1, 0].tick_params(axis='x', rotation=45)

        # Plot 4: Hits, Stands, Splits
        self.axs[1, 1].bar(action_data.keys(), action_data.values())
        self.axs[1, 1].set_title("Actions")
        self.axs[1, 1].set_ylabel("Count")
        self.axs[1, 1].tick_params(axis='x', rotation=45)

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2, wspace=0.4, hspace=0.8)

        # Refresh the canvas
        self.canvas.draw()

    def log_message(self, message):
        self.game_log.append(message)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.yview(tk.END)

    def start_game(self):
        self.hand_count += 1
        self.deal_card(self.player_hands[0])
        self.deal_card(self.dealer_hand, is_dealer=True)
        self.deal_card(self.player_hands[0])
        self.deal_card(self.dealer_hand, is_dealer=True)

        player_value = self.calculate_hand_value(self.player_hands[0])
        dealer_value = self.calculate_hand_value(self.dealer_hand)
        if player_value == 21:
            reward = self.available_rewards["Blackjack"]
            self.blackjack_wins += 1
            self.update_game_label(reward)
            self.returns.append(reward)
            self.average_returns.append(sum(self.returns) / self.hand_count)
            message = f"{self.log_prefix(0)} Blackjack! Player wins ${reward}!"
            self.log_message(message)
            self.reset_hand()
        elif dealer_value == 21:
            reward = self.available_rewards["Loss"]
            self.losses += 1
            self.update_game_label(reward)
            self.returns.append(reward)
            self.average_returns.append(sum(self.returns) / self.hand_count)
            message = f"{self.log_prefix(0)} Dealer Hit Blackjack. Player Loses ${-reward}"
            self.log_message(message)
            self.reset_hand()
        else:
            self.update_game_label(0)

        # Enable the split button if the player has two cards of the same rank
        self.update_split_button_state()

    def update_split_button_state(self):
        can_split = any(len(hand) == 2 and hand[0][0] == hand[1][0] for hand in self.player_hands)
        self.split_button.config(state=tk.NORMAL if can_split else tk.DISABLED)

    def split(self):
        # Check if the active hand can be split
        active_hand = self.player_hands[self.active_hand]
        if len(active_hand) == 2 and active_hand[0][0] == active_hand[1][0]:
            self.splits += 1
            new_hand = [self.player_hands[self.active_hand].pop()]
            self.deal_card(new_hand)
            self.deal_card(self.player_hands[self.active_hand])
            self.player_hands.append(new_hand)

        self.update_split_button_state()
        for frame in self.player_frames[1:]:
            frame.pack(side=tk.LEFT, padx=10)

        self.update_hand_display()
        self.update_game_label(0)
        self.update_split_button_state()

    def highlight_active_hand(self):
        for idx, player_frame in enumerate(self.player_frames):
            player_frame.config(bg='lightblue' if self.active_hand == idx else 'SystemButtonFace')

        self.update_split_button_state()

    def update_hand_display(self):
        for player_frame in self.player_frames:
            for widget in player_frame.winfo_children():
                widget.destroy()

        for idx, player_hand in enumerate(self.player_hands):
            for card in player_hand:
                img_path = f'./data/test/{card[0]} of {card[1]}/1.jpg'
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        img = img.resize((100, 150), Image.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        self.card_images.append(photo)
                        frame = tk.Frame(self.player_frames[idx])
                        frame.pack(side=tk.LEFT)
                        label = tk.Label(frame, image=photo)
                        label.image = photo
                        label.pack()
                        tk.Label(frame, text=f'{card[0]} of {card[1]}').pack()
                    except Exception as e:
                        print(f"Failed to load image {img_path}: {e}")

        self.highlight_active_hand()

    def hit(self):
        self.deal_card(self.player_hands[self.active_hand])
        self.hits += 1
        player_value = self.calculate_hand_value(self.player_hands[self.active_hand])
        if player_value >= 21:
            self.stands -= 1
            self.stand()

        self.update_game_label(0)
        self.update_split_button_state()

    def compare_hands(self, player_value, dealer_value, hand_idx):
        player_blackjack_win = player_value == 21
        player_regular_win = (player_value < 21 and (dealer_value > 21 or player_value > dealer_value))
        draw = player_value < 21 and player_value == dealer_value
        dealer_wins = (dealer_value <= 21 and player_value < dealer_value) or player_value > 21

        message = ""
        if player_blackjack_win:
            reward = self.available_rewards["Blackjack"]
            self.blackjack_wins += 1
            self.update_game_label(reward)
            self.returns.append(reward)
            message = f"{self.log_prefix(hand_idx)} Blackjack! Player wins ${reward}!"
        elif player_regular_win:
            reward = self.available_rewards["Regular"]
            self.regular_wins += 1
            self.update_game_label(reward)
            self.returns.append(reward)

            if player_value > dealer_value:
                message += f"{self.log_prefix(hand_idx)} Player wins ${reward}! Value: {player_value} > Dealer Value: {dealer_value}"
            elif dealer_value > 21:
                message += f"{self.log_prefix(hand_idx)} Player wins ${reward}! Dealer Busts with {dealer_value}"

        elif draw:
            self.draws += 1
            self.update_game_label(0)
            self.returns.append(0)
            message += f"{self.log_prefix(hand_idx)} Draw! Value: {player_value} == Dealer Value: {dealer_value}"
        elif dealer_wins:
            reward = self.available_rewards["Loss"]
            self.losses += 1
            self.update_game_label(reward)
            self.returns.append(reward)

            if player_value < dealer_value:
                message += f"{self.log_prefix(hand_idx)} Player loses ${-reward}. Dealer Value: {dealer_value} > Value: {player_value}"
            elif player_value > 21:
                message += f"{self.log_prefix(hand_idx)} Player loses ${-reward}. Player Busts! {player_value}"

        self.log_message(message)

    def stand(self):
        self.stands += 1
        if self.active_hand < len(self.player_hands) - 1:
            self.active_hand += 1
            self.highlight_active_hand()
            if self.active_hand < len(self.player_hands):
                return

        # If no more hands to play, process the dealer's turn
        while self.calculate_hand_value(self.dealer_hand) < 17:
            self.deal_card(self.dealer_hand, is_dealer=True)

        dealer_value = self.calculate_hand_value(self.dealer_hand)
        player_values = [self.calculate_hand_value(hand) for hand in self.player_hands]

        # Compare dealer's hand with player's hands
        for idx, player_value in enumerate(player_values):
            self.compare_hands(player_value, dealer_value, idx)

        self.reset_hand()

    def reset_hand(self):
        if self.hand_count != 0:
            self.average_returns.append(sum(self.returns) / self.hand_count)
            messagebox.showinfo("Blackjack", f'Hand {self.hand_count} completed')

        # Clear and reinitialize player frames
        for player_frame in self.player_frames:
            for widget in player_frame.winfo_children():
                widget.destroy()

            player_frame.pack_forget()

        self.player_frames = [tk.Frame(self.frames_container, padx=25) for _ in range(4 * self.card_counter.n_decks)]
        for frame in self.player_frames:
            frame.pack(side=tk.LEFT)

        # Clear dealer frame
        for widget in self.dealer_frame.winfo_children():
            widget.destroy()

        # Reinitialize hands and variables
        self.player_hands = [[]]
        self.dealer_hand = []
        self.active_hand = 0

        # Restart the game
        self.start_game()
        self.enable_buttons()

    def disable_buttons(self):
        self.hit_button.config(state=tk.DISABLED)
        self.stand_button.config(state=tk.DISABLED)

    def enable_buttons(self):
        self.hit_button.config(state=tk.NORMAL)
        self.stand_button.config(state=tk.NORMAL)

    def reset_game(self):
        message = f'Game Reset after {self.hand_count} hands\nAverage return: ${sum(self.returns) / self.hand_count} / hand\nTotal return: ${self.bankroll - self.starting_bankroll}, {self.blackjack_wins} Blackjacks, {self.regular_wins} Regular Wins, {self.losses} Losses, {self.draws} Draws\n{self.hits} Hits, {self.splits} Splits, {self.stands} Stands'
        self.log_message(message)
        messagebox.showinfo("Blackjack", message)
        self.deck = self.create_deck()
        self.card_counter.reset_deck()
        self.bankroll = self.starting_bankroll
        self.regular_wins = 0
        self.blackjack_wins = 0
        self.draws = 0
        self.losses = 0
        self.hits = 0
        self.splits = 0
        self.stands = 0
        self.returns = []
        self.average_returns = []
        self.hand_count = 0
        self.update_game_label(0)
        self.card_images = []
        self.probabilities = {'High': [5 / 13], 'Low': [5 / 13], 'Neutral': [3 / 13]}
        self.reset_hand()


class BlackjackEnvironment:
    def __init__(self, model_path: str):
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

        # Load the trained model
        self.model_path = model_path
        self.model = self.load_model()

        # Initialize the card counter
        self.card_counter = CardCounter(model=self.model)
        self.deck = self.create_deck()

        self._blackjack_wins = 0
        self._regular_wins = 0
        self._losses = 0
        self._draws = 0
        self._hits = 0
        self._splits = 0
        self._stands = 0
        self._hand_count = 0
        self._active_hand = 0
        self._dealer_hand = []
        self._player_hands = [[]]
        self._dealer_hand_real = []
        self._player_hands_real = [[]]
        self.reset_deck()

    @property
    def blackjack_wins(self):
        return self._blackjack_wins

    @blackjack_wins.setter
    def blackjack_wins(self, blackjack_wins):
        self._blackjack_wins = blackjack_wins

    @property
    def regular_wins(self):
        return self._regular_wins

    @regular_wins.setter
    def regular_wins(self, regular_wins):
        self._regular_wins = regular_wins

    @property
    def losses(self):
        return self._losses

    @losses.setter
    def losses(self, losses):
        self._losses = losses

    @property
    def draws(self):
        return self._draws

    @draws.setter
    def draws(self, draws):
        self._draws = draws

    @property
    def hits(self):
        return self._hits

    @hits.setter
    def hits(self, hits):
        self._hits = hits

    @property
    def splits(self):
        return self._splits

    @splits.setter
    def splits(self, splits):
        self._splits = splits

    @property
    def stands(self):
        return self._stands

    @stands.setter
    def stands(self, stands):
        self._stands = stands

    @property
    def hand_count(self):
        return self._hand_count

    @hand_count.setter
    def hand_count(self, hand_count):
        self._hand_count = hand_count

    @property
    def active_hand(self):
        return self._active_hand

    @active_hand.setter
    def active_hand(self, active_hand):
        self._active_hand = active_hand

    @property
    def dealer_hand(self):
        return self._dealer_hand

    @dealer_hand.setter
    def dealer_hand(self, dealer_hand):
        self._dealer_hand = dealer_hand

    @property
    def player_hands(self):
        return self._player_hands

    @player_hands.setter
    def player_hands(self, player_hands):
        self._player_hands = player_hands

    @property
    def dealer_hand_real(self):
        return self._dealer_hand_real

    @dealer_hand_real.setter
    def dealer_hand_real(self, dealer_hand_real):
        self._dealer_hand_real = dealer_hand_real

    @property
    def player_hands_real(self):
        return self._player_hands_real

    @player_hands_real.setter
    def player_hands_real(self, player_hands_real):
        self._player_hands_real = player_hands_real

    def load_model(self, use_kan: bool = True):
        input_shape = (224, 224, 3)
        num_classes = 53  # 53 classes for 53 cards

        model = KANCNN(input_shape, num_classes) if use_kan else Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
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
        random.shuffle(deck)
        return deck

    def deal_card(self):
        card = self.deck.pop()
        self.card_counter.total_cards -= 1
        if self.card_counter.total_cards == 0:
            self.deck = self.create_deck()
            self.card_counter.reset_deck()

        img_path = f'./data/test/{card[0]} of {card[1]}/1.jpg'
        images = [tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)]
        prob_high, prob_low, prob_neutral, predicted_cards = self.card_counter.count_cards(images=images)
        predicted_card = predicted_cards[0]

        return card, predicted_card

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

    def reset(self):
        self.player_hands = [[]]
        self.player_hands_real = [[]]
        self.dealer_hand_real = []
        self.dealer_hand = []
        self.active_hand = 0
        
        first_player_card, first_player_predicted_card = self.deal_card()
        first_dealer_card, first_dealer_predicted_card = self.deal_card()
        second_player_card, second_player_predicted_card = self.deal_card()
        second_dealer_card, second_dealer_predicted_card = self.deal_card()
        
        self.player_hands[self.active_hand].append(first_player_predicted_card)
        self.player_hands_real[self.active_hand].append(first_player_card)
        self.dealer_hand.append(first_dealer_predicted_card)
        self.dealer_hand_real.append(first_dealer_card)
        self.player_hands[self.active_hand].append(second_player_predicted_card)
        self.player_hands_real[self.active_hand].append(second_player_card)
        self.dealer_hand.append(second_dealer_predicted_card)
        self.dealer_hand_real.append(second_dealer_card)
        
        return self.get_state()

    def reset_deck(self):
        self.deck = self.create_deck()
        self.card_counter.reset_deck()
        self.probabilities = {'High': [5 / 13], 'Low': [5 / 13], 'Neutral': [3 / 13]}

    def get_state(self):
        player_hand_value = self.calculate_hand_value(self.player_hands[self.active_hand])
        dealer_upcard = self.ranks[self.dealer_hand[0][0]]
        usable_a = self.usable_ace(self.player_hands[self.active_hand])
        can_split = len(self.player_hands_real[self.active_hand]) == 2 and self.player_hands_real[self.active_hand][0][0] == self.player_hands_real[self.active_hand][1][0]
        return (player_hand_value, dealer_upcard, usable_a, len(self.player_hands), can_split)

    def usable_ace(self, hand):
        return any(card[0] == 'ace' for card in hand) and self.calculate_hand_value(hand) + 10 <= 21

    def step(self, action):
        if action == 'hit':
            self.hits += 1
            card, predicted_card = self.deal_card()
            self.player_hands_real[self.active_hand].append(card)
            self.player_hands[self.active_hand].append(predicted_card)
            player_value = self.calculate_hand_value(self.player_hands_real[self.active_hand])
            if player_value > 21:
                self.losses += 1
                reward = -1
                done = True
                return self.get_state(), reward, done
            if player_value == 21:
                self.blackjack_wins += 1
                reward = 1.5
                done = True
                return self.get_state(), reward, done
            else:
                return self.get_state(), 0, False

        elif action == 'stand':
            self.stands += 1
            if self.active_hand < len(self.player_hands) - 1:
                self.active_hand += 1
                reward = 0
                done = False
                return self.get_state(), reward, done

            while self.calculate_hand_value(self.dealer_hand_real) < 17:
                card, predicted_card = self.deal_card()
                self.dealer_hand.append(predicted_card)
                self.dealer_hand_real.append(card)

            dealer_value = self.calculate_hand_value(self.dealer_hand_real)
            rewards = 0
            for hand in self.player_hands_real:
                player_value = self.calculate_hand_value(hand)
                if player_value == 21:
                    rewards += 1.5
                    self.blackjack_wins += 1
                elif player_value < dealer_value <= 21:
                    rewards -= 1
                    self.losses += 1
                elif dealer_value > 21 or player_value > dealer_value:
                    rewards += 1
                    self.regular_wins += 1
                elif player_value == dealer_value and player_value < 21:
                    self.draws += 1

            done = True
            return self.get_state(), rewards, done

        elif action == 'split' and len(self.player_hands_real[self.active_hand]) == 2 and self.player_hands_real[self.active_hand][0][0] == self.player_hands_real[self.active_hand][1][0]:
            self.splits += 1
            first_card, first_predicted_card = self.deal_card()
            second_card, second_predicted_card = self.deal_card()
            new_hand = [self.player_hands[self.active_hand].pop()]
            new_real_hand = [self.player_hands_real[self.active_hand].pop()]

            self.player_hands_real[self.active_hand].append(first_card)
            self.player_hands[self.active_hand].append(first_predicted_card)
            new_hand.append(second_predicted_card)
            new_real_hand.append(second_card)

            self.player_hands_real.append(new_real_hand)
            self.player_hands.append(new_hand)
            reward = 0
            done = False
            return self.get_state(), reward, done

        return self.get_state(), 0, False


class QLearningAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(len(action_space)))
        self.actions = action_space
        self.rewards = []
        self.action_counts = defaultdict(int)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            available_actions = self.actions if state[4] else [a for a in self.actions if a != 'split']
            action = np.random.choice(available_actions)
        else:
            q_values = self.q_table[state]
            if not state[4]:
                self.q_table[state][self.actions.index('split')] = -float('inf')

            action = self.actions[np.argmax(q_values)]

        self.action_counts[action] += 1
        return action

    def update_q_value(self, state, action, reward, next_state):
        if not state[4]:
            self.q_table[state][self.actions.index('split')] = -float('inf')

        if not next_state[4]:
            self.q_table[next_state][self.actions.index('split')] = -float('inf')

        action_idx = self.actions.index(action)
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        self.q_table[state][action_idx] += self.alpha * (td_target - self.q_table[state][action_idx])

    def train(self, environment, episodes):
        for episode in range(episodes):
            state = environment.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = environment.step(action)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            self.rewards.append(total_reward)

        return self.rewards

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)


class BlackjackRL:
    def __init__(self, root, agent, environment, train_episodes, test_episodes, cnn_model_path):
        self.root = root
        self.root.title("Blackjack RL Agent")
        self.agent = agent
        self.environment = environment
        self.train_episodes = train_episodes
        self.test_episodes = test_episodes
        self._probabilities = {'High': [5 / 13], 'Low': [5 / 13], 'Neutral': [3 / 13]}
        self.card_images = []
        self.game_log = []
        self.training = False

        # Get screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Set window size as a percentage of screen size
        window_width = int(screen_width * 0.95)
        window_height = int(screen_height * 0.9)
        self.root.geometry(f"{window_width}x{window_height}")

        # Calculate padding as percentages of screen dimensions
        self.padx = int(screen_width * 0.01)
        self.pady = int(screen_height * 0.01)
        self.num_columns = 4

        # Load the trained model
        self.cnn_model_path = cnn_model_path
        self.cnn_model = self.load_cnn_model()
        self.card_counter = CardCounter(model=self.cnn_model)

        # Create frames
        self.game_frame = tk.Frame(self.root)
        self.game_frame.pack(side=tk.LEFT, padx=self.padx, pady=self.pady, expand=True, fill='both')
        self.stats_frame = tk.Frame(self.root)
        self.stats_frame.pack(side=tk.RIGHT, padx=self.padx, pady=self.pady, expand=True, fill='both')

        # Create game widgets
        self.player_label = tk.Label(self.game_frame, text="Player's Hand")
        self.player_label.grid(row=0, column=0, columnspan=self.num_columns)
        self.player_frames = [tk.Frame(self.game_frame) for _ in range(4)]
        for idx, frame in enumerate(self.player_frames):
            frame.grid(row=1, column=idx, padx=self.padx, pady=self.pady, sticky='nsew')

        self.dealer_label = tk.Label(self.game_frame, text="Dealer's Hand")
        self.dealer_label.grid(row=2, column=0, columnspan=self.num_columns, pady=(self.pady * 2, 0))
        self.dealer_frame = tk.Frame(self.game_frame)
        self.dealer_frame.grid(row=3, column=0, columnspan=self.num_columns, padx=self.padx, pady=self.pady, sticky='nsew')

        self.start_button = tk.Button(self.game_frame, text="Start Training", command=self.start_training)
        self.start_button.grid(row=4, column=0, padx=self.padx, pady=self.pady)
        self.stop_button = tk.Button(self.game_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.grid(row=4, column=1, padx=self.padx, pady=self.pady)
        self.save_button = tk.Button(self.game_frame, text="Save Model", command=self.save_rl_model)
        self.save_button.grid(row=5, column=0, padx=self.padx, pady=self.pady)
        self.load_button = tk.Button(self.game_frame, text="Load Model", command=self.load_rl_model)
        self.load_button.grid(row=5, column=1, padx=self.padx, pady=self.pady)

        self.test_button = tk.Button(self.game_frame, text="Test Agent", command=lambda: self.test_agent(games=test_episodes), state=tk.DISABLED)
        self.test_button.grid(row=6, column=0, columnspan=self.num_columns, pady=self.pady)

        # Create stats widgets with adjusted sizes
        fig_width = window_width / 3 / 100
        fig_height = window_height / 3 / 100

        # Create stats widgets
        self.cumulative_rewards_fig, self.cumulative_rewards_ax = plt.subplots(figsize=(fig_width, fig_height))
        self.cumulative_rewards_canvas = FigureCanvasTkAgg(self.cumulative_rewards_fig, master=self.stats_frame)
        self.cumulative_rewards_canvas.get_tk_widget().grid(row=0, column=0)

        self.average_rewards_fig, self.average_rewards_ax = plt.subplots(figsize=(fig_width, fig_height))
        self.average_rewards_canvas = FigureCanvasTkAgg(self.average_rewards_fig, master=self.stats_frame)
        self.average_rewards_canvas.get_tk_widget().grid(row=0, column=1)

        self.win_loss_fig, self.win_loss_ax = plt.subplots(figsize=(fig_width, fig_height))
        self.win_loss_canvas = FigureCanvasTkAgg(self.win_loss_fig, master=self.stats_frame)
        self.win_loss_canvas.get_tk_widget().grid(row=1, column=0)

        self.prob_fig, self.prob_ax = plt.subplots(figsize=(fig_width, fig_height))
        self.prob_canvas = FigureCanvasTkAgg(self.prob_fig, master=self.stats_frame)
        self.prob_canvas.get_tk_widget().grid(row=1, column=1)

        self.q_values_fig, self.q_values_ax = plt.subplots(figsize=(fig_width, fig_height))
        self.q_values_canvas = FigureCanvasTkAgg(self.q_values_fig, master=self.stats_frame)
        self.q_values_canvas.get_tk_widget().grid(row=2, column=0)

        self.action_counts_fig, self.action_counts_ax = plt.subplots(figsize=(fig_width, fig_height))
        self.action_counts_canvas = FigureCanvasTkAgg(self.action_counts_fig, master=self.stats_frame)
        self.action_counts_canvas.get_tk_widget().grid(row=2, column=1)

        # Create a scrollable text widget for logs
        self.log_frame = tk.Frame(self.game_frame)
        self.log_frame.grid(row=7, column=0, columnspan=2, sticky='nsew')
        self.log_text = tk.Text(self.log_frame, wrap=tk.WORD, state=tk.NORMAL, width=80, height=10)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_scrollbar = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=self.log_scrollbar.set)

        self.episode_count = 0

    @property
    def probabilities(self):
        return self._probabilities

    @probabilities.setter
    def probabilities(self, probabilities):
        self._probabilities = probabilities

    def load_cnn_model(self, use_kan: bool = True):
        input_shape = (224, 224, 3)
        num_classes = 53  # 53 classes for 53 cards

        model = KANCNN(input_shape, num_classes) if use_kan else Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.load_weights(self.cnn_model_path)  # Load your trained model weights
        return model

    def save_rl_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("H5 files", "*.h5"), ("PKL files", "*.pkl"), ("All files", "*.*")])
        if file_path:
            self.agent.save_model(file_path)
            print(f"Model saved to: {file_path}")

    def load_rl_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5"), ("PKL files", "*.pkl"), ("All files", "*.*")])
        if file_path:
            self.agent.load_model(file_path)
            print(f"Model loaded from: {file_path}")

    def display_hand(self, hand, frame, is_dealer=True):
        for widget in frame.winfo_children():
            widget.destroy()

        for i, card in enumerate(hand):
            img_path = f'./data/test/{card[0]} of {card[1]}/1.jpg'

            # Debugging: Check if the image path exists
            if not os.path.exists(img_path):
                print(f"Image path does not exist: {img_path}")
                continue

            images = [tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)]
            prob_high, prob_low, prob_neutral, predicted_cards = self.card_counter.count_cards(images=images)
            predicted_card = predicted_cards[0]
            predicted_img_path = f'./data/test/{predicted_card[0]} of {predicted_card[1]}/1.jpg'

            if not os.path.exists(predicted_img_path):
                print(f"Image path does not exist: {predicted_img_path}")
                return

            try:
                img = Image.open(img_path)
                img = img.resize((100, 150), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.card_images.append(photo)  # Keep a reference to the image
                label = tk.Label(frame, image=photo)
                label.image = photo  # Ensure reference is kept by the label
                label.grid(row=0, column=i)
                frame_text = f'{card[0]} of {card[1]}'
                if is_dealer:
                    frame_text = f'Actual: {frame_text}\nPredicted: {predicted_card[0]} of {predicted_card[1]}'

                tk.Label(frame, text=frame_text).grid(row=1, column=i)
            except Exception as e:
                print(f"Failed to load image {img_path}: {e}")

        return hand

    def update_stats(self):
        # Update cumulative rewards plot
        cumulative_rewards = np.cumsum(self.agent.rewards)
        self.cumulative_rewards_ax.clear()
        self.cumulative_rewards_ax.plot(range(1, len(cumulative_rewards) + 1), cumulative_rewards, label='Cumulative Reward')
        self.cumulative_rewards_ax.set_xlabel('Episode')
        self.cumulative_rewards_ax.set_ylabel('Cumulative Reward')
        self.cumulative_rewards_ax.set_title('Cumulative Rewards Over Time')
        self.cumulative_rewards_ax.legend()
        self.cumulative_rewards_canvas.draw()

        # Update average rewards plot
        average_rewards = cumulative_rewards / (np.arange(len(self.agent.rewards)) + 1)
        self.average_rewards_ax.clear()
        self.average_rewards_ax.plot(range(1, len(average_rewards) + 1), average_rewards, label='Average Reward per Episode')
        self.average_rewards_ax.set_xlabel('Episode')
        self.average_rewards_ax.set_ylabel('Average Reward')
        self.average_rewards_ax.set_title('Average Rewards Over Time')
        self.average_rewards_ax.legend()
        self.average_rewards_canvas.draw()

        # Update win/loss plot
        win_data = {
            "Regular Wins": self.environment.regular_wins,
            "Blackjack Wins": self.environment.blackjack_wins,
            "Losses": self.environment.losses,
            "Draws": self.environment.draws
        }
        self.win_loss_ax.clear()
        self.win_loss_ax.bar(win_data.keys(), win_data.values())
        self.win_loss_ax.set_title("Wins, Losses, Draws")
        self.win_loss_ax.set_ylabel("Count")
        self.win_loss_canvas.draw()

        # Update probability plot
        prob_data = {
            "High": self.probabilities["High"][-1],
            "Low": self.probabilities["Low"][-1],
            "Neutral": self.probabilities["Neutral"][-1]
        }
        self.prob_ax.clear()
        self.prob_ax.bar(prob_data.keys(), prob_data.values())
        self.prob_ax.set_title("Card Probabilities")
        self.prob_ax.set_ylabel("Count")
        self.prob_canvas.draw()

        # Update Q-values plot
        state = self.environment.get_state()
        q_values = self.agent.q_table[state]
        if not state[4]:
            q_values[self.agent.actions.index('split')] = 0

        self.q_values_ax.clear()
        self.q_values_ax.bar(self.agent.actions, q_values)
        self.q_values_ax.set_xlabel('Actions')
        self.q_values_ax.set_ylabel('Q-values')
        self.q_values_ax.set_title('Q-values for Current State')
        self.q_values_canvas.draw()

        # Update action counts plot
        actions = list(self.agent.action_counts.keys())
        counts = list(self.agent.action_counts.values())
        self.action_counts_ax.clear()
        self.action_counts_ax.bar(actions, counts)
        self.action_counts_ax.set_xlabel('Actions')
        self.action_counts_ax.set_ylabel('Counts')
        self.action_counts_ax.set_title('Action Selection Frequency')
        self.action_counts_canvas.draw()

    def log_message(self, message):
        self.game_log.append(message)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.yview(tk.END)

    def start_training(self):
        self.training = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.test_button.config(state=tk.DISABLED)
        self.train_agent()

    def stop_training(self):
        self.training = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.test_button.config(state=tk.NORMAL)

    def reset_game(self):
        self.environment.reset()
        for frame in self.player_frames:
            for widget in frame.winfo_children():
                widget.destroy()

        for widget in self.dealer_frame.winfo_children():
            widget.destroy()

        self.player_frames = [tk.Frame(self.game_frame) for _ in range(4)]
        for idx, frame in enumerate(self.player_frames):
            frame.grid(row=1, column=idx, padx=self.padx, pady=self.pady, sticky='nsew')

        self.dealer_frame = tk.Frame(self.game_frame)
        self.dealer_frame.grid(row=3, column=0, columnspan=self.num_columns, padx=self.padx, pady=self.pady, sticky='nsew')

        self.update_stats()

    def train_agent(self):
        if not self.training:
            return

        self.reset_game()
        state = self.environment.reset()
        done = False
        while not done:
            self.root.update_idletasks()
            self.root.update()
            action = self.agent.choose_action(state)
            next_state, reward, done = self.environment.step(action)
            self.agent.update_q_value(state, action, reward, next_state)
            state = next_state
            self.update_hand_display()
            if done:
                self.episode_count += 1
                self.agent.rewards.append(reward)
                self.update_stats()
                self.log_message(f"Episode {self.episode_count}: Reward {reward}")
                self.root.after(1000, self.train_agent)
                break

            self.root.after(1000, lambda: None)  # Add delay for visualization

    def update_hand_display(self):
        for idx, hand in enumerate(self.environment.player_hands_real):
            self.display_hand(hand, self.player_frames[idx])

        self.display_hand(self.environment.dealer_hand_real, self.dealer_frame, is_dealer=True)

    def test_agent(self, games=100):
        wins, losses, draws = 0, 0, 0
        for _ in range(games):
            state = self.environment.reset()
            done = False
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done = self.environment.step(action)
                state = next_state
                if done:
                    if reward == 1:
                        wins += 1
                    elif reward == -1:
                        losses += 1
                    else:
                        draws += 1

        messagebox.showinfo("Test Results", f"Results over {games} games:\nWins: {wins}\nLosses: {losses}\nDraws: {draws}")

    def start(self):
        self.root.mainloop()


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, state_size, action_space, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=0.001, batch_size=32):
        self.state_size = state_size
        self.action_size = len(action_space)
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = DQN(state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.rewards = []
        self.action_counts = defaultdict(int)

    def remember(self, state, action, reward, next_state, done):
        action_idx = self.action_space.index(action)
        self.memory.append((state, action_idx, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            available_actions = self.action_space if state[-1] else [a for a in self.action_space if a != 'split']
            return random.choice(available_actions)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        if not state[-1]:
            act_values[self.action_space.index('split')] = -float('inf')

        action_idx = torch.argmax(act_values).item()
        self.action_counts[self.action_space[action_idx]] += 1
        return self.action_space[action_idx]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action_idx, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state)
                target += self.gamma * torch.max(self.model(next_state)).item()
            state = torch.FloatTensor(state)
            target_f = self.model(state).detach()
            target_f[action_idx] = target
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, name):
        self.model.load_state_dict(torch.load(name))

    def save_model(self, name):
        torch.save(self.model.state_dict(), name)


class DQNBlackjackEnvironment:
    def __init__(self, model_path):
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

        # Load the trained model
        self.model_path = model_path
        self.model = self.load_model()

        # Initialize the card counter
        self.card_counter = CardCounter(model=self.model)
        self.deck = self.create_deck()

        self._blackjack_wins = 0
        self._regular_wins = 0
        self._losses = 0
        self._draws = 0
        self._hits = 0
        self._splits = 0
        self._stands = 0
        self._hand_count = 0
        self._active_hand = 0
        self._dealer_hand = []
        self._player_hands = [[]]
        self._dealer_hand_real = []
        self._player_hands_real = [[]]
        self.reset_deck()

    @property
    def blackjack_wins(self):
        return self._blackjack_wins

    @blackjack_wins.setter
    def blackjack_wins(self, blackjack_wins):
        self._blackjack_wins = blackjack_wins

    @property
    def regular_wins(self):
        return self._regular_wins

    @regular_wins.setter
    def regular_wins(self, regular_wins):
        self._regular_wins = regular_wins

    @property
    def losses(self):
        return self._losses

    @losses.setter
    def losses(self, losses):
        self._losses = losses

    @property
    def draws(self):
        return self._draws

    @draws.setter
    def draws(self, draws):
        self._draws = draws

    @property
    def hits(self):
        return self._hits

    @hits.setter
    def hits(self, hits):
        self._hits = hits

    @property
    def splits(self):
        return self._splits

    @splits.setter
    def splits(self, splits):
        self._splits = splits

    @property
    def stands(self):
        return self._stands

    @stands.setter
    def stands(self, stands):
        self._stands = stands

    @property
    def hand_count(self):
        return self._hand_count

    @hand_count.setter
    def hand_count(self, hand_count):
        self._hand_count = hand_count

    @property
    def active_hand(self):
        return self._active_hand

    @active_hand.setter
    def active_hand(self, active_hand):
        self._active_hand = active_hand

    @property
    def dealer_hand(self):
        return self._dealer_hand

    @dealer_hand.setter
    def dealer_hand(self, dealer_hand):
        self._dealer_hand = dealer_hand

    @property
    def player_hands(self):
        return self._player_hands

    @player_hands.setter
    def player_hands(self, player_hands):
        self._player_hands = player_hands

    @property
    def dealer_hand_real(self):
        return self._dealer_hand_real

    @dealer_hand_real.setter
    def dealer_hand_real(self, dealer_hand_real):
        self._dealer_hand = dealer_hand_real

    @property
    def player_hands_real(self):
        return self._player_hands_real

    @player_hands_real.setter
    def player_hands_real(self, player_hands_real):
        self._player_hands_real = player_hands_real

    def load_model(self, use_kan: bool = True):
        input_shape = (224, 224, 3)
        num_classes = 53  # 53 classes for 53 cards

        model = KANCNN(input_shape, num_classes) if use_kan else Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
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
        random.shuffle(deck)
        return deck

    def deal_card(self):
        card = self.deck.pop()
        self.card_counter.total_cards -= 1
        if self.card_counter.total_cards == 0:
            self.deck = self.create_deck()
            self.card_counter.reset_deck()

        img_path = f'./data/test/{card[0]} of {card[1]}/1.jpg'
        images = [tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)]
        prob_high, prob_low, prob_neutral, predicted_cards = self.card_counter.count_cards(images=images)
        predicted_card = predicted_cards[0]

        return card, predicted_card

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

    def reset(self):
        self.player_hands = [[]]
        self.player_hands_real = [[]]
        self.dealer_hand_real = []
        self.dealer_hand = []
        self.active_hand = 0

        first_player_card, first_player_predicted_card = self.deal_card()
        first_dealer_card, first_dealer_predicted_card = self.deal_card()
        second_player_card, second_player_predicted_card = self.deal_card()
        second_dealer_card, second_dealer_predicted_card = self.deal_card()

        self.player_hands[self.active_hand].append(first_player_predicted_card)
        self.player_hands_real[self.active_hand].append(first_player_card)
        self.dealer_hand.append(first_dealer_predicted_card)
        self.dealer_hand_real.append(first_dealer_card)
        self.player_hands[self.active_hand].append(second_player_predicted_card)
        self.player_hands_real[self.active_hand].append(second_player_card)
        self.dealer_hand.append(second_dealer_predicted_card)
        self.dealer_hand_real.append(second_dealer_card)

        return self.get_state()

    def reset_deck(self):
        self.deck = self.create_deck()
        self.card_counter.reset_deck()
        self.probabilities = {'High': [5 / 13], 'Low': [5 / 13], 'Neutral': [3 / 13]}

    def one_hot_encode_rank(self, rank):
        ranks = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'jack', 'queen', 'king', 'ace']
        encoding = [0] * len(ranks)
        encoding[ranks.index(rank)] = 1
        return encoding

    def get_state(self):
        player_hand_value = self.calculate_hand_value(self.player_hands[self.active_hand])
        dealer_visible_card = self.one_hot_encode_rank(self.dealer_hand[0][0])
        usable_a = self.usable_ace(self.player_hands[self.active_hand])
        return [player_hand_value] + dealer_visible_card + [usable_a] + [self.probabilities['High'][-1], self.probabilities['Low'][-1], self.probabilities['Neutral'][-1], self.can_split()]

    def can_split(self):
        return len(self.player_hands_real[self.active_hand]) == 2 and self.player_hands_real[self.active_hand][0][0] == self.player_hands_real[self.active_hand][1][0]

    def usable_ace(self, hand):
        return 1 in [card[0] == 'ace' for card in hand] and self.calculate_hand_value(hand) + 10 <= 21

    def step(self, action):
        if action == 'hit':
            self.hits += 1
            card, predicted_card = self.deal_card()
            self.player_hands_real[self.active_hand].append(card)
            self.player_hands[self.active_hand].append(predicted_card)
            player_value = self.calculate_hand_value(self.player_hands_real[self.active_hand])
            if player_value > 21:
                self.losses += 1
                reward = -1
                done = True
                return self.get_state(), reward, done
            elif player_value == 21:
                self.blackjack_wins += 1
                reward = 1.5
                done = True
                return self.get_state(), reward, done
            else:
                return self.get_state(), 0, False

        elif action == 'stand':
            self.stands += 1
            if self.active_hand < len(self.player_hands) - 1:
                self.active_hand += 1
                reward = 0
                done = False
                return self.get_state(), reward, done

            while self.calculate_hand_value(self.dealer_hand_real) < 17:
                card, predicted_card = self.deal_card()
                self.dealer_hand.append(predicted_card)
                self.dealer_hand_real.append(card)

            dealer_value = self.calculate_hand_value(self.dealer_hand_real)
            rewards = 0
            for hand in self.player_hands_real:
                player_value = self.calculate_hand_value(hand)
                if player_value == 21:
                    rewards += 1.5
                    self.blackjack_wins += 1
                elif player_value < dealer_value <= 21:
                    rewards -= 1
                    self.losses += 1
                elif dealer_value > 21 or player_value > dealer_value:
                    rewards += 1
                    self.regular_wins += 1
                elif player_value == dealer_value and player_value < 21:
                    self.draws += 1

            done = True
            return self.get_state(), rewards, done

        elif action == 'split' and len(self.player_hands_real[self.active_hand]) == 2 and self.player_hands_real[self.active_hand][0][0] == self.player_hands_real[self.active_hand][1][0]:
            self.splits += 1
            first_card, first_predicted_card = self.deal_card()
            second_card, second_predicted_card = self.deal_card()
            new_hand = [self.player_hands[self.active_hand].pop()]
            new_real_hand = [self.player_hands_real[self.active_hand].pop()]

            self.player_hands_real[self.active_hand].append(first_card)
            self.player_hands[self.active_hand].append(first_predicted_card)
            new_hand.append(second_predicted_card)
            new_real_hand.append(second_card)

            self.player_hands_real.append(new_real_hand)
            self.player_hands.append(new_hand)
            reward = 0
            done = False
            return self.get_state(), reward, done

        return self.get_state(), 0, False


class DQNBlackjackRL:
    def __init__(self, root, agent, environment, train_episodes, test_episodes, cnn_model_path):
        self.root = root
        self.root.title("Blackjack DQN RL Agent")
        self.agent = agent
        self.environment = environment
        self.train_episodes = train_episodes
        self.test_episodes = test_episodes
        self._probabilities = {'High': [5 / 13], 'Low': [5 / 13], 'Neutral': [3 / 13]}
        self.card_images = []
        self.game_log = []
        self.training = False

        # Get screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Set window size as a percentage of screen size
        window_width = int(screen_width * 0.95)
        window_height = int(screen_height * 0.9)
        self.root.geometry(f"{window_width}x{window_height}")

        # Calculate padding as percentages of screen dimensions
        self.padx = int(screen_width * 0.01)
        self.pady = int(screen_height * 0.01)
        self.num_columns = 4

        # Load the trained model
        self.cnn_model_path = cnn_model_path
        self.cnn_model = self.load_cnn_model()
        self.card_counter = CardCounter(model=self.cnn_model)

        # Create frames
        self.game_frame = tk.Frame(self.root)
        self.game_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.stats_frame = tk.Frame(self.root)
        self.stats_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Create frames
        self.game_frame = tk.Frame(self.root)
        self.game_frame.pack(side=tk.LEFT, padx=self.padx, pady=self.pady, expand=True, fill='both')
        self.stats_frame = tk.Frame(self.root)
        self.stats_frame.pack(side=tk.RIGHT, padx=self.padx, pady=self.pady, expand=True, fill='both')

        # Create game widgets
        self.player_label = tk.Label(self.game_frame, text="Player's Hand")
        self.player_label.grid(row=0, column=0, columnspan=self.num_columns)
        self.player_frames = [tk.Frame(self.game_frame) for _ in range(4)]
        for idx, frame in enumerate(self.player_frames):
            frame.grid(row=1, column=idx, padx=self.padx, pady=self.pady, sticky='nsew')

        self.dealer_label = tk.Label(self.game_frame, text="Dealer's Hand")
        self.dealer_label.grid(row=2, column=0, columnspan=self.num_columns, pady=(self.pady * 2, 0))
        self.dealer_frame = tk.Frame(self.game_frame)
        self.dealer_frame.grid(row=3, column=0, columnspan=self.num_columns, padx=self.padx, pady=self.pady, sticky='nsew')

        self.start_button = tk.Button(self.game_frame, text="Start Training", command=self.start_training)
        self.start_button.grid(row=4, column=0, padx=self.padx, pady=self.pady)
        self.stop_button = tk.Button(self.game_frame, text="Stop Training", command=self.stop_training,
                                     state=tk.DISABLED)
        self.stop_button.grid(row=4, column=1, padx=self.padx, pady=self.pady)
        self.save_button = tk.Button(self.game_frame, text="Save Model", command=self.save_rl_model)
        self.save_button.grid(row=5, column=0, padx=self.padx, pady=self.pady)
        self.load_button = tk.Button(self.game_frame, text="Load Model", command=self.load_rl_model)
        self.load_button.grid(row=5, column=1, padx=self.padx, pady=self.pady)

        self.test_button = tk.Button(self.game_frame, text="Test Agent",
                                     command=lambda: self.test_agent(games=test_episodes), state=tk.DISABLED)
        self.test_button.grid(row=6, column=0, columnspan=self.num_columns, pady=self.pady)

        # Create stats widgets with adjusted sizes
        fig_width = window_width / 3 / 100
        fig_height = window_height / 3 / 100

        # Create stats widgets
        self.cumulative_rewards_fig, self.cumulative_rewards_ax = plt.subplots(figsize=(fig_width, fig_height))
        self.cumulative_rewards_canvas = FigureCanvasTkAgg(self.cumulative_rewards_fig, master=self.stats_frame)
        self.cumulative_rewards_canvas.get_tk_widget().grid(row=0, column=0)

        self.average_rewards_fig, self.average_rewards_ax = plt.subplots(figsize=(fig_width, fig_height))
        self.average_rewards_canvas = FigureCanvasTkAgg(self.average_rewards_fig, master=self.stats_frame)
        self.average_rewards_canvas.get_tk_widget().grid(row=0, column=1)

        self.win_loss_fig, self.win_loss_ax = plt.subplots(figsize=(fig_width, fig_height))
        self.win_loss_canvas = FigureCanvasTkAgg(self.win_loss_fig, master=self.stats_frame)
        self.win_loss_canvas.get_tk_widget().grid(row=1, column=0)

        self.prob_fig, self.prob_ax = plt.subplots(figsize=(fig_width, fig_height))
        self.prob_canvas = FigureCanvasTkAgg(self.prob_fig, master=self.stats_frame)
        self.prob_canvas.get_tk_widget().grid(row=1, column=1)

        self.q_values_fig, self.q_values_ax = plt.subplots(figsize=(fig_width, fig_height))
        self.q_values_canvas = FigureCanvasTkAgg(self.q_values_fig, master=self.stats_frame)
        self.q_values_canvas.get_tk_widget().grid(row=2, column=0)

        self.action_counts_fig, self.action_counts_ax = plt.subplots(figsize=(fig_width, fig_height))
        self.action_counts_canvas = FigureCanvasTkAgg(self.action_counts_fig, master=self.stats_frame)
        self.action_counts_canvas.get_tk_widget().grid(row=2, column=1)

        # Create a scrollable text widget for logs
        self.log_frame = tk.Frame(self.game_frame)
        self.log_frame.grid(row=7, column=0, columnspan=2, sticky='nsew')
        self.log_text = tk.Text(self.log_frame, wrap=tk.WORD, state=tk.NORMAL, width=80, height=10)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_scrollbar = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=self.log_scrollbar.set)

        self.episode_count = 0

    @property
    def probabilities(self):
        return self._probabilities

    @probabilities.setter
    def probabilities(self, probabilities):
        self._probabilities = probabilities

    def load_cnn_model(self, use_kan: bool = True):
        input_shape = (224, 224, 3)
        num_classes = 53  # 53 classes for 53 cards

        model = KANCNN(input_shape, num_classes) if use_kan else Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.load_weights(self.cnn_model_path)  # Load your trained model weights
        return model

    def save_rl_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".h5",
                                                 filetypes=[("H5 files", "*.h5"), ("PKL files", "*.pkl"),
                                                            ("All files", "*.*")])
        if file_path:
            self.agent.save_model(file_path)
            print(f"Model saved to: {file_path}")

    def load_rl_model(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("H5 files", "*.h5"), ("PKL files", "*.pkl"), ("All files", "*.*")])
        if file_path:
            self.agent.load_model(file_path)
            print(f"Model loaded from: {file_path}")

    def display_hand(self, hand, frame, is_dealer=True):
        for widget in frame.winfo_children():
            widget.destroy()

        for i, card in enumerate(hand):
            img_path = f'./data/test/{card[0]} of {card[1]}/1.jpg'

            # Debugging: Check if the image path exists
            if not os.path.exists(img_path):
                print(f"Image path does not exist: {img_path}")
                continue

            images = [tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)]
            prob_high, prob_low, prob_neutral, predicted_cards = self.card_counter.count_cards(images=images)
            predicted_card = predicted_cards[0]
            predicted_img_path = f'./data/test/{predicted_card[0]} of {predicted_card[1]}/1.jpg'

            if not os.path.exists(predicted_img_path):
                print(f"Image path does not exist: {predicted_img_path}")
                return

            try:
                img = Image.open(img_path)
                img = img.resize((100, 150), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.card_images.append(photo)  # Keep a reference to the image
                label = tk.Label(frame, image=photo)
                label.image = photo  # Ensure reference is kept by the label
                label.grid(row=0, column=i)
                frame_text = f'{card[0]} of {card[1]}'
                if is_dealer:
                    frame_text = f'Actual: {frame_text}\nPredicted: {predicted_card[0]} of {predicted_card[1]}'

                tk.Label(frame, text=frame_text).grid(row=1, column=i)
            except Exception as e:
                print(f"Failed to load image {img_path}: {e}")

        return hand

    def update_stats(self):
        # Update cumulative rewards plot
        cumulative_rewards = np.cumsum(self.agent.rewards)
        self.cumulative_rewards_ax.clear()
        self.cumulative_rewards_ax.plot(range(1, len(cumulative_rewards) + 1), cumulative_rewards, label='Cumulative Reward')
        self.cumulative_rewards_ax.set_xlabel('Episode')
        self.cumulative_rewards_ax.set_ylabel('Cumulative Reward')
        self.cumulative_rewards_ax.set_title('Cumulative Rewards Over Time')
        self.cumulative_rewards_ax.legend()
        self.cumulative_rewards_canvas.draw()

        # Update average rewards plot
        average_rewards = cumulative_rewards / (np.arange(len(self.agent.rewards)) + 1)
        self.average_rewards_ax.clear()
        self.average_rewards_ax.plot(range(1, len(average_rewards) + 1), average_rewards, label='Average Reward per Episode')
        self.average_rewards_ax.set_xlabel('Episode')
        self.average_rewards_ax.set_ylabel('Average Reward')
        self.average_rewards_ax.set_title('Average Rewards Over Time')
        self.average_rewards_ax.legend()
        self.average_rewards_canvas.draw()

        # Update win/loss plot
        win_data = {
            "Regular Wins": self.environment.regular_wins,
            "Blackjack Wins": self.environment.blackjack_wins,
            "Losses": self.environment.losses,
            "Draws": self.environment.draws
        }
        self.win_loss_ax.clear()
        self.win_loss_ax.bar(win_data.keys(), win_data.values())
        self.win_loss_ax.set_title("Wins, Losses, Draws")
        self.win_loss_ax.set_ylabel("Count")
        self.win_loss_canvas.draw()

        # Update probability plot
        prob_data = {
            "High": self.probabilities["High"][-1],
            "Low": self.probabilities["Low"][-1],
            "Neutral": self.probabilities["Neutral"][-1]
        }
        self.prob_ax.clear()
        self.prob_ax.bar(prob_data.keys(), prob_data.values())
        self.prob_ax.set_title("Card Probabilities")
        self.prob_ax.set_ylabel("Count")
        self.prob_canvas.draw()

        # Update Q-values plot
        state = self.environment.get_state()
        q_values = self.agent.model(torch.FloatTensor(state)).detach().numpy()
        if not state[-1]:
            q_values[self.agent.action_space.index('split')] = 0

        self.q_values_ax.clear()
        self.q_values_ax.bar(self.agent.action_space, q_values)
        self.q_values_ax.set_xlabel('Actions')
        self.q_values_ax.set_ylabel('Q-values')
        self.q_values_ax.set_title('Q-values for Current State')
        self.q_values_canvas.draw()

        # Update action counts plot
        actions = list(self.agent.action_counts.keys())
        counts = list(self.agent.action_counts.values())
        self.action_counts_ax.clear()
        self.action_counts_ax.bar(actions, counts)
        self.action_counts_ax.set_xlabel('Actions')
        self.action_counts_ax.set_ylabel('Counts')
        self.action_counts_ax.set_title('Action Selection Frequency')
        self.action_counts_canvas.draw()

    def log_message(self, message):
        self.game_log.append(message)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.yview(tk.END)

    def start_training(self):
        self.training = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.test_button.config(state=tk.DISABLED)
        self.train_agent()

    def stop_training(self):
        self.training = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.test_button.config(state=tk.NORMAL)

    def reset_game(self):
        self.environment.reset()
        for frame in self.player_frames:
            for widget in frame.winfo_children():
                widget.destroy()

        for widget in self.dealer_frame.winfo_children():
            widget.destroy()

        self.player_frames = [tk.Frame(self.game_frame) for _ in range(4)]
        for idx, frame in enumerate(self.player_frames):
            frame.grid(row=1, column=idx, padx=self.padx, pady=self.pady, sticky='nsew')

        self.dealer_frame = tk.Frame(self.game_frame)
        self.dealer_frame.grid(row=3, column=0, columnspan=self.num_columns, padx=self.padx, pady=self.pady,
                               sticky='nsew')

        self.update_stats()

    def train_agent(self):
        if not self.training:
            return

        self.reset_game()
        state = self.environment.reset()
        done = False
        while not done:
            self.root.update_idletasks()
            self.root.update()
            action = self.agent.act(state)
            next_state, reward, done = self.environment.step(action)
            self.agent.remember(state, action, reward, next_state, done)
            self.agent.replay()
            state = next_state
            self.update_hand_display()
            if done:
                self.episode_count += 1
                self.agent.rewards.append(reward)
                self.update_stats()
                self.log_message(f"Episode {self.episode_count}: Reward {reward}")
                self.root.after(1000, self.train_agent)
                break

            self.root.after(1000, lambda: None)  # Add delay for visualization

    def update_hand_display(self):
        for idx, hand in enumerate(self.environment.player_hands_real):
            self.display_hand(hand, self.player_frames[idx])

        self.display_hand(self.environment.dealer_hand_real, self.dealer_frame, is_dealer=True)

    def test_agent(self, games=100):
        wins, losses, draws = 0, 0, 0
        for _ in range(games):
            state = self.environment.reset()
            done = False
            while not done:
                action = self.agent.act(state)
                next_state, reward, done = self.environment.step(action)
                state = next_state
                if done:
                    if reward == 1:
                        wins += 1
                    elif reward == -1:
                        losses += 1
                    else:
                        draws += 1

        messagebox.showinfo("Test Results",
                            f"Results over {games} games:\nWins: {wins}\nLosses: {losses}\nDraws: {draws}")

    def start(self):
        self.root.mainloop()


if __name__ == '__main__':
    epochs = 5
    model_path = f'./models/{epochs}_model.h5'

    # Train the CNN model
    model = train_model(epochs=epochs)

    # Make predictions
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

    # Create a blackjack game
    root = tk.Tk()
    game = BlackjackGame(root, 18000, f'./models/{epochs}_model.h5')
    game.start_game()
    root.mainloop()

    # Create a Q-Learning Reinforcement Learning model to learn to play blackjack
    env = BlackjackEnvironment(model_path)
    actions = ['hit', 'stand', 'split']
    agent = QLearningAgent(action_space=actions)
    root = tk.Tk()
    app = BlackjackRL(
        root=root,
        agent=agent,
        environment=env,
        train_episodes=1000,
        test_episodes=100,
        cnn_model_path=model_path
    )

    app.start()

    # Create a Deep Q Learning model to learn to play blackjack
    # Calculate the state size based on the returned state from get_state method
    player_hand_value_size = 1
    dealer_visible_card_size = 13  # One-hot encoding of 13 ranks
    usable_ace_size = 1
    probabilities_size = 3
    split_possibility_indicator = 1

    state_size = player_hand_value_size + dealer_visible_card_size + usable_ace_size + probabilities_size + split_possibility_indicator
    actions = ['hit', 'stand', 'split']

    dqn_env = DQNBlackjackEnvironment(model_path=model_path)
    dqn_agent = DQNAgent(state_size=state_size, action_space=actions)

    # Create and start the GUI
    dqn_root = tk.Tk()
    dqn_app = DQNBlackjackRL(
        root=dqn_root,
        agent=dqn_agent,
        environment=dqn_env,
        train_episodes=1000,
        test_episodes=100,
        cnn_model_path=model_path
    )

    dqn_app.start()

import pygame
import sys
import numpy as np
import pandas as pd
from scipy.io import arff
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# === Load EEG test set ===
file_path = r"C:\Users\Lenovo\Documents\EEG Eye State.arff"
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
df['eyeDetection'] = df['eyeDetection'].astype(int)

channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 
            'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

X = df[channels].values
y = df['eyeDetection'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
y_test = y

# === Load LSTM model ===
model = load_model(r"D:\app_gui_als\EEG_LSTM_Model.h5")
print("Model loaded successfully!")

# === Pygame setup ===
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 400
FPS = 60
GRAVITY = 0.8
JUMP_STRENGTH = -15
OBSTACLE_WIDTH = 40
OBSTACLE_HEIGHT = 60
OBSTACLE_SPEED = 5
NUM_LIVES = 40
REACTION_DISTANCE = 80  # react only when obstacle is very close
WIN_SCORE = 100
SCORE_PER_OBSTACLE = 5

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("EEG Jump Game")
clock = pygame.time.Clock()

# Load player sprite
player_sprite = pygame.image.load("sprite.png").convert_alpha()
player_sprite = pygame.transform.scale(player_sprite, (50, 50))

# === Player setup ===
player_x = 100
player_y = SCREEN_HEIGHT - 60
player_vel_y = 0
on_ground = True
lives = NUM_LIVES
score = 0

# === Obstacle class ===
class Obstacle:
    def __init__(self, x, y, width, height, speed):
        self.rect = pygame.Rect(x, y, width, height)
        self.speed = speed
        self.scored = False

    def update(self):
        self.rect.x -= self.speed

    def draw(self, surface):
        pygame.draw.rect(surface, (255, 0, 0), self.rect)

# === Obstacles setup ===
obstacles = []
obstacle_timer = 0
OBSTACLE_INTERVAL = 90  # frames

# === Pointer for EEG test set ===
test_index = 0

# === Game loop ===
running = True
while running:
    screen.fill((135, 206, 235))  # Sky blue background

    # Draw player
    screen.blit(player_sprite, (player_x, player_y))

    # Spawn obstacles
    obstacle_timer += 1
    if obstacle_timer >= OBSTACLE_INTERVAL:
        obstacle_timer = 0
        obstacles.append(Obstacle(SCREEN_WIDTH, SCREEN_HEIGHT - OBSTACLE_HEIGHT,
                                  OBSTACLE_WIDTH, OBSTACLE_HEIGHT, OBSTACLE_SPEED))

    # Move and draw obstacles
    for obs in obstacles[:]:
        obs.update()
        obs.draw(screen)

        # Check collision
        if pygame.Rect(player_x, player_y, 50, 50).colliderect(obs.rect):
            lives -= 1
            obstacles.remove(obs)
            if lives <= 0:
                print("Game Over!")
                running = False

        # Check if obstacle passed player successfully
        elif obs.rect.right < player_x and not obs.scored:
            score += SCORE_PER_OBSTACLE
            obs.scored = True
            obstacles.remove(obs)  # remove obstacle after scoring
            if score >= WIN_SCORE:
                print("You Win!")
                running = False

        # Remove off-screen obstacles
        elif obs.rect.right < 0:
            obstacles.remove(obs)

    # Player gravity
    if not on_ground:
        player_vel_y += GRAVITY
        player_y += player_vel_y
        if player_y >= SCREEN_HEIGHT - 60:
            player_y = SCREEN_HEIGHT - 60
            player_vel_y = 0
            on_ground = True

    # === Feed EEG row to model only when obstacle is very close ===
    if obstacles and test_index < len(X_test):
        next_obstacle = obstacles[0]
        distance_to_obstacle = next_obstacle.rect.x - (player_x + 30)  # distance from player right edge

        if distance_to_obstacle <= REACTION_DISTANCE:
            sample_input = X_test[test_index].reshape(1, 1, 14)
            true_label = y_test[test_index]
            pred_prob = model.predict(sample_input, verbose=0)[0][0]

            # Jump only if true label = 1 and model predicts > 0.5
            if true_label == 1 and pred_prob > 0.5 and on_ground:
                player_vel_y = JUMP_STRENGTH
                on_ground = False

            test_index += 1

    # Display lives and score
    font = pygame.font.SysFont(None, 36)
    lives_text = font.render(f"Lives: {lives}", True, (0, 0, 0))
    score_text = font.render(f"Score: {score}", True, (0, 0, 0))
    screen.blit(lives_text, (10, 10))
    screen.blit(score_text, (SCREEN_WIDTH - 150, 10))  # score on right side

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()
    clock.tick(FPS)

pygame.quit()
sys.exit()

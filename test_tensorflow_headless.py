import random
import retro
import numpy as np
import cv2
import tensorflow as tf
import pickle

zones = ['AngelIslandZone.Act1', 'CarnivalNightZone.Act1', 'DeathEggZone.Act1', 'DeathEggZone.Act2', 'FlyingBatteryZone.Act1', 'HydrocityZone.Act1', 'IcecapZone.Act2', 'LaunchBaseZone.Act1', 'LavaReefZone.Act2', 'MarbleGardenZone.Act2', 'FlyingBatteryZone.Act2', 'SandopolisZone.Act1']

env = retro.make(game="SonicAndKnuckles3-Genesis", state=random.choice(zones), players=1)
imgarray = []

# Define the network architecture
input_size = 320 * 224
hidden_size = 256
output_size = env.action_space.n
num_layers = 32
resume = False
def create_network():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(hidden_size, input_shape=(input_size,), activation='relu'))

    for _ in range(num_layers - 2):
        model.add(tf.keras.layers.Dense(hidden_size, activation='relu'))

    model.add(tf.keras.layers.Dense(output_size, activation='softmax'))

    return model

# Load the pre-trained network model or create a new one
model_path = 'network_model.h5'
if resume:
    network = tf.keras.models.load_model(model_path)
else:
    network = create_network()

# Randomly initialize the network
for layer in network.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        weights = layer.get_weights()
        weights = [np.random.randn(*w.shape) for w in weights]
        layer.set_weights(weights)

# Define the optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

while True:
    #zone = random.choice(zones)
    zone = "AngelIslandZone.Act1"
    env.load_state(zone)
    if zone == "AngelIslandZone.Act1":
        offset = 5001
    elif zone == "HydrocityZone.Act1":
        offset = 639
    elif zone == "MarbleGardenZone.Act2":
        offset = 95
    elif zone == "CarnivalNightZone.Act1":
        offset = 157
    elif zone == "IcecapZone.Act2":
        offset = 223
    elif zone == "LaunchBaseZone.Act1":
        offset = 175
    elif zone == "MushroomHillZone.Act2":
        offset = 335
    elif zone == "FlyingBatteryZone.Act1":
        offset = 95
    elif zone == "FlyingBatteryZone.Act2":
        offset = 95
    elif zone == "SandopolisZone.Act1":
        offset = 191
    elif zone == "LavaReefZone.Act2":
        offset = 223
    elif zone == "DeathEggZone.Act1":
        offset = 1314
    else:
        offset = 319
    ob = env.reset()

    inx, iny, inc = env.observation_space.shape
    #inx = int(32)        #Old code to set the
    #iny = int(32)        #input size to 32x32

    current_max_fitness = 0
    fitness_current = 0
    frame = 0
    kickt = 120
    rings = 0
    xpos2 = 0
    score = 0
    if zone == "IcecapZone.Act2" or zone == "LavaReefZone.Act2":
        counter = 240
        rrwd = 1
    else:
        counter = 180
        rrwd = 0
    xpos = 0
    done = False

    while not done:
        frame += 1
        kickt -= 1
        ob = cv2.resize(ob, (inx, iny))
        ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
        ob = np.reshape(ob, (inx, iny))
        imgarray = np.ndarray.flatten(ob)

        with tf.GradientTape() as tape:
            nnOutput = network(np.array([imgarray]))
            action = nnOutput[0]
            ob, rew, done, info = env.step(action)

            xpos = info['x']
            rings = info['rings']
            if rrwd == 0 and rings >= 25:
                print("Well done, 25 rings!")
                rrwd = 1000
            if xpos >= 12216:
                fitness_current += 10000000
                done = True

            fitness_current = (info['x'] - offset) + (rings * 10) + rrwd

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 120
            else:
                counter -= 1

            if kickt == 0 and info['x'] <= offset:
                current_max_fitness += -10000000
                fitness_current += -1000000
                done = True

            if done or counter == 0:
                done = True
                print("Fitness: " + str(fitness_current) + "\nRings: " + str(
                    info['rings']) + ", Score: " + str(info['score']) + "\nPOS - offsets: " + str(
                    (info['x'] - offset)) + " X=" + str(info['x']) + " in " + str(zone))

        # Compute gradients and update weights
        gradients = tape.gradient(nnOutput, network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, network.trainable_variables))

    # Save the network model
    network.save(model_path)
    print(f"Network model saved to {model_path}")

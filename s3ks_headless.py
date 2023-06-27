import random
import retro
import numpy as np
import cv2
import neat
import pickle
zones = ['AngelIslandZone.Act1', 'CarnivalNightZone.Act1', 'DeathEggZone.Act1', 'DeathEggZone.Act2', 'FlyingBatteryZone.Act1', 'HydrocityZone.Act1', 'IcecapZone.Act2', 'LaunchBaseZone.Act1', 'LavaReefZone.Act2', 'MarbleGardenZone.Act2', 'FlyingBatteryZone.Act2', 'SandopolisZone.Act1']
env = retro.make(game = "SonicAndKnuckles3-Genesis", state = random.choice(zones), players=1)
imgarray = []
xpos_end = 0

resume = False
restore_file = "neat-checkpoint-57"
def eval_genomes(genomes, config):


    for genome_id, genome in genomes:
#       zone = random.choice(zones)
        zone = "AngelIslandZone.Act1"
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
        env.load_state(zone)
        ob = env.reset()
        ac = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape
        inx = int(32)
        iny = int(32)
#        inx = int(inx/8)
#        iny = int(iny/8)
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        
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
            
            #env.render()
            frame += 1
            kickt -= 1
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx,iny))
            imgarray = np.ndarray.flatten(ob)

            nnOutput = net.activate(imgarray)
            
            ob, rew, done, info = env.step(nnOutput)
            
            xpos2 = info['p2x']
            xpos = info['x']
            rings = info['rings']
            if rrwd == 0 and rings >= 25:
                    print("Well done, 25 rings!")
                    rrwd = 1000
            if xpos >= 9767:
                    fitness_current += 10000000
                    done = True
            #fitness_current += rew
            fitness_current = (info['x'] - offset) + (rings * 10) + rrwd # just trying
            #print(fitness_current,info['x']) # viewing variables
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
                print("ID: " + str(genome_id) + ", Fitness: " + str(fitness_current) + "\nRings: " + str(info['rings']) + ", Score: " + str(info['score']) + "\nPOS - offsets: " + str((info['x'] - offset)) + " X=" + str(info['x']) + " in " + str(zone))
            genome.fitness = fitness_current

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-why')

if resume == True:
    p = neat.Checkpointer.restore_checkpoint(restore_file)
else:
    p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(1))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

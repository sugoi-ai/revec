"""
Adapted from: https://github.com/ikostrikov/pytorch-a3c/blob/master/main.py
The main file needed within a3c. Runs of the train and test functions from their respective files.
Example of use:
`cd algorithms/a3c`
`python main.py`

Runs A3C on our AI2ThorEnv wrapper with default params (4 processes). Optionally it can be
run on any atari environment as well using the --atari and --atari-env-name params.
"""
import sys
import argparse
import os
import numpy as np
import torch
import torch.multiprocessing as mp
from multiprocessing import Manager
import json
import h5py

sys.path.append('..') # to access env package

from env.ai2thor_env import AI2ThorDumpEnv
from optimizers import SharedAdam, SharedRMSprop
from model import ActorCritic
from test import test, live_test, test_multi
from train import train, train_multi

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--about', type=str, default="training A3C", required=True,
                    help='description about training, also the name of saving directory, \
                            just a way to control which test case was run')
parser.add_argument('--lr', type=float, default=7e-4,
                    help='learning rate (default: 7e-4)')
parser.add_argument('--angle', type=float, default=45.0,
                    help='rotation angle')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.96,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--ec', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--vc', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max_grad_norm', type=float, default=10,
                    help='value loss coefficient (default: 10)')
parser.add_argument('--lr_decay', type=int, default=0,
                    help='whether to use learning rate decay')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--room_id', type=int, default=0,
                    help='room id (default: 0)')
parser.add_argument('--test', type=int, default=0,
                    help='whether to activate testing phase')
parser.add_argument('--live_test', type=int, default=0,
                    help='whether to activate live testing phase')
parser.add_argument('--action_size', type=int, default=4,
                    help='number of possible actions')
parser.add_argument('--num_processes', type=int, default=20,
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num_iters', type=int, default=100,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--num_epochs', type=int, default=20000,
                    help='number of epochs to run on each thread')
parser.add_argument('--max_episode_length', type=int, default=1000,
                    help='maximum length of an episode (default: 1000)')
parser.add_argument('--siamese', type=int, default=0)
parser.add_argument('--train_cnn', type=int, default=0,
                    help='whether to re-train cnn module')
parser.add_argument('--history_size', type=int, default=4,
                    help='whether to stack frames')
parser.add_argument('--optim', type=int, default=1,
                    help='optimizer: 0 for Adam, 1 for RMSprop')
parser.add_argument('--multi_scene', type=int, default=1,
                    help='whether to train on multiple scenes')
parser.add_argument('--lstm', type=int, default=0,
                    help='whether to use lstm instead of stacking features')
parser.add_argument('--onehot', type=int, default=1,
                    help='whether to use onehot vector as input feature')
parser.add_argument('--embed', type=int, default=1,
                    help='embedding mode: 0 for onehot, 1 for fasttext')
parser.add_argument('--random_test', type=int, default=0,
                    help='whether to test performance of a random agent')
parser.add_argument('--use_gcn', type=int, default=0,
                    help='whether to include gcn')
parser.add_argument('--anti_col', type=int, default=0,
                    help='whether to include collision penalty to rewarding scheme')
parser.add_argument('--use_graph', type=int, default=0,
                    help='whether to use relations vector from graph to replace gcn')
parser.add_argument('--yolo_gcn', type=int, default=0,
                    help='whether to use yolo as input for gcn instead of resnet')
parser.add_argument('--no_shared', type=int, default=0,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--scene_id', type=int, default=1,
                    help='scene id (default: 1)')
parser.add_argument('--gpu_ids', type=int, default=-1,
                    nargs='+', help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument('--hard', type=int, default=1,
                    help='whether to make environment harder\
                        0: agent only has to reach the correct position\
                        1: agent has to reach the correct position and has right rotation')

parser.add_argument('--config_file', type=str, default="../config.json")
parser.add_argument('--folder', type=str, default=None)

ALL_ROOMS = {
    0: "Kitchens",
    1: "Living Rooms",
    2: "Bedrooms",
    3: "Bathrooms"
}

def read_config(config_path):
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
    return config

def read_weights(folder):
    weights = [f for f in os.listdir(folder) if f.endswith('.pth')]
    histories = [f for f in os.listdir(folder) if f.endswith('.pkl')]

    arguments = json.load(open(folder + '/arguments.json'))
    if arguments['multi_scene']:        
        scenes = list(set([f.split('_')[0] for f in os.listdir(folder) if f.endswith('.pkl')]))
        targets = []
    else:
        scenes = ["FloorPlan{}".format(arguments['scene_id'])]        
        targets = list(set([f.split('_')[1] for f in os.listdir(folder) if f.endswith('.pkl')]))
    
    print(list(zip(range(len(weights)), weights)))
    wid = input("Please specify weights: ")
    weights = weights[int(wid)]

    return os.path.join(folder, weights), arguments, {'scenes': scenes, 'targets': targets}

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)

    config = read_config(args.config_file)
    # room = config['rooms'][ALL_ROOMS[args.room_id]]

    if args.folder is not None:
        weights, arguments, info = read_weights(args.folder)

        multi_scene =  len(info['scenes']) > 1

        if args.test or args.live_test:
            if not args.random_test:
                shared_model = ActorCritic(config, arguments)
                shared_model.share_memory()

                shared_model.load_state_dict(torch.load(weights, map_location='cpu'))
                print("loaded model")
            else:
                print("testing random agent ..")
                shared_model = None

    if args.live_test:
        assert args.folder is not None
        print("Start testing ..")

        if multi_scene:
            print(list(zip(range(len(info['scenes'])), info['scenes'])))
            command = int(input("Please specify scene id. \nYour input:"))
            training_scene = info['scenes'][command]

            f = h5py.File("{}.hdf5".format(os.path.join(config['dump_path'], training_scene)), 'r')
            # training_objects = f['all_visible_objects'][()].tolist()
            training_objects = ["GarbageCan", "Sink", "Bread", "StoveKnob", "SinkBasin", "StoveBurner", "Fridge", "CounterTop", "Microwave", "LightSwitch", "CoffeeMachine", "Cabinet"]
            f.close()
        else:
            # training_scene = info['scenes'][0]y
            # print(list(zip(range(len(info['targets'])), info['targets'])))
            # command = input("Please specify target ids, you can choose either individually (e.g: 0,1,2) or by range (e.g: 0-4)\nYour input:")
            #n
            # if '-' not in command:
            #     target_ids = [int(i.strip()) for i in command.split(",")]
            # else:
            #     target_ids = list(range(int(command.split('-')[0]), int(command.split('-')[1]) + 1))
            #
            # training_objects = [info['targets'][target_id] for target_id in target_ids]
            command = input("Please specify scene id. \nYour input:")
            training_scene = command

            training_objects = ["Desk","GarbageCan", "Sink", "Bread", "StoveKnob", "SinkBasin", "StoveBurner", "Fridge", "CounterTop", "Microwave", "LightSwitch", "CoffeeMachine", "Cabinet"]
        # training_objects.sort()
        live_test('FloorPlan'+training_scene, training_objects, shared_model, config, arguments)

    else:
        if args.test:
            assert args.folder is not None
            if not multi_scene:
                
                training_scene = info['scenes'][0]
                testing_objects = config["picked"][training_scene]['test']
                training_objects = info['targets']
                all_visible_objects =  training_objects + testing_objects

                phase = ['train'] * len(training_objects) + ['test'] * len(testing_objects)

                print(list(zip(range(len(phase)), all_visible_objects, phase)))
                command = input("Please specify target ids, you can choose either individually (e.g: 0,1,2) or by range (e.g: 0-4)\nYour input:")
                if '-' not in command:
                    target_ids = [int(i.strip()) for i in command.split(",")]
                else:
                    target_ids = list(range(int(command.split('-')[0]), int(command.split('-')[1]) + 1))

                chosen_objects = [all_visible_objects[target_id] for target_id in target_ids]
                check_phase = lambda c: 'train' if os.path.isfile(os.path.join(args.folder, "net_{}.pth".format(c))) else 'test'
                chosen_phases  = [check_phase(c) for c in chosen_objects]

                results = mp.Array('f', len(chosen_objects))
                processes = []
                for rank, obj in enumerate(chosen_objects):
                    p = mp.Process(target=test, args=(training_scene, obj, rank, shared_model, \
                                    results, config, arguments))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

                print("Testing accuracies:", list(zip(chosen_objects, chosen_phases, results[:])))

            else:
                arguments['test'] = 1
                
                print(list(zip(range(len(ALL_ROOMS)), list(ALL_ROOMS.values()))))
                command = input("Please specify room type:")
                scene_type = ALL_ROOMS[int(command)]

                training_scenes = config['rooms'][scene_type]['train_scenes']
                testing_scenes = config['rooms'][scene_type]['test_scenes']

                command = input("Training/testing scenes. (0, 1): ")
                scenes = [training_scenes, testing_scenes][int(command)]

                results = Manager().dict()

                all_visible_objects = config['rooms'][scene_type]['train_objects'] + config['rooms'][scene_type]['test_objects']
                chosen_phases = ['train'] * len(config['rooms'][scene_type]['train_objects']) + ['test'] * len(config['rooms'][scene_type]['test_objects'])
                for obj in all_visible_objects:
                    results[obj] = []

                processes = []

                counter = mp.Value('i', 0)
                lock = mp.Lock()

                for rank in range(0, len(scenes)):
                    p = mp.Process(target=test_multi, args=(scenes[rank], rank, shared_model, \
                                    results, config, arguments))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()        

                accuracies = []
                avg_sc = {'train': [], 'test': []}
                avg_spl = {'train': [], 'test': []}
                for obj in all_visible_objects:
                    accuracies.append((np.mean(results[obj]), np.mean(np.array(results[obj], dtype=bool))))

                for phase, acc in zip(chosen_phases, accuracies):
                    avg_spl[phase].append(acc[0])
                    avg_sc[phase].append(acc[1])

                avg_sc['train'] = np.mean(avg_sc['train'])
                avg_sc['test'] = np.mean(avg_sc['test'])
                avg_spl['train'] = np.mean(avg_spl['train'])
                avg_spl['test'] = np.mean(avg_spl['test'])

                print("Accuracies:", list(zip(all_visible_objects, chosen_phases, accuracies)))    
                print("[Avergae] SPL: {} | SR: {}".format(avg_spl, avg_sc))

        else:
            arguments = vars(args)
            weights = None

            if not arguments['multi_scene']:
        
                if not os.path.isdir("training-history/{}".format(arguments['about'])):
                    os.mkdir("training-history/{}".format(arguments['about']))

                with open('training-history/{}/arguments.json'.format(arguments['about']), 'w') as outfile:
                    json.dump(arguments, outfile)
                
                training_scene = "FloorPlan{}".format(arguments['scene_id'])
                f = h5py.File("{}.hdf5".format(os.path.join(config['dump_path'], training_scene)), 'r')
                all_visible_objects = f['all_visible_objects'][()].tolist()
                f.close()
                
                testing_objects = config["picked"][training_scene]['test']
                trainable_objects = list(set(all_visible_objects) - set(testing_objects))
            
                print(list(zip(range(len(trainable_objects)), trainable_objects)))
                command = input("Please specify target ids, you can choose either individually (e.g: 0,1,2) or by range (e.g: 0-4)\nYour input:")
                if '-' not in command:
                    target_ids = [int(i.strip()) for i in command.split(",")]
                else:
                    target_ids = list(range(int(command.split('-')[0]), int(command.split('-')[1]) + 1))

                training_objects = [trainable_objects[target_id] for target_id in target_ids]
                num_thread_each = arguments['num_processes'] // len(training_objects)
                object_threads = []

                for obj in training_objects:
                    object_threads += [obj] * num_thread_each

                object_threads += [np.random.choice(training_objects)] * (arguments['num_processes'] - len(object_threads))

                print("Start training agent to find {} in {}".format(training_objects, training_scene))

                shared_model = ActorCritic(config, arguments)
                shared_model.share_memory()

                if weights is not None:
                    shared_model.load_state_dict(torch.load(weights, map_location='cpu'))
                    print("loaded model")

                scheduler = None
                if arguments['no_shared']:
                    optimizer = None
                else:
                    if arguments['optim'] == 0:
                        optimizer = SharedAdam(shared_model.parameters(), lr=arguments['lr'])
                    else:
                        optimizer = SharedRMSprop(shared_model.parameters(), lr=arguments['lr'], alpha=0.99, eps=0.1)

                    optimizer.share_memory()
                    if arguments['lr_decay']:
                        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99995)

                processes = []

                counter = mp.Value('i', 0)
                lock = mp.Lock()

                for rank in range(0, arguments['num_processes']):
                    p = mp.Process(target=train, args=(training_scene, object_threads[rank], rank, shared_model, \
                                    scheduler, counter, lock, config, arguments, optimizer))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

            else:

                print(list(zip(range(len(ALL_ROOMS)), list(ALL_ROOMS.values()))))
                command = input("Please specify room type:")
                scene_type = ALL_ROOMS[int(command)]

                training_scenes = config['rooms'][scene_type]['train_scenes']
                num_thread_each = arguments['num_processes'] // len(training_scenes)
                scene_threads = []

                for s in training_scenes:
                    scene_threads += [s] * num_thread_each

                scene_threads += list(np.random.choice(training_scenes, arguments['num_processes'] - len(scene_threads)))

                if not os.path.isdir("training-history/{}".format(arguments['about'])):
                    os.mkdir("training-history/{}".format(arguments['about']))

                with open('training-history/{}/arguments.json'.format(arguments['about']), 'w') as outfile:
                    json.dump(arguments, outfile)

                    print("Start training agent in {}".format(training_scenes))

                shared_model = ActorCritic(config, arguments)
                shared_model.share_memory()

                if weights is not None:
                    shared_model.load_state_dict(torch.load(weights, map_location='cpu'))
                    print("loaded model")

                scheduler = None
                if arguments['no_shared']:
                    optimizer = None
                else:
                    if arguments['optim'] == 0:
                        optimizer = SharedAdam(shared_model.parameters(), lr=arguments['lr'])
                    else:
                        optimizer = SharedRMSprop(shared_model.parameters(), lr=arguments['lr'], alpha=0.99, eps=0.1)

                    optimizer.share_memory()
                    if arguments['lr_decay']:
                        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99995)

                processes = []

                counter = mp.Value('i', 0)
                lock = mp.Lock()

                for rank in range(0, arguments['num_processes']):
                    p = mp.Process(target=train_multi, args=(scene_threads[rank], rank, shared_model, \
                                    scheduler, counter, lock, config, arguments, optimizer))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()                            


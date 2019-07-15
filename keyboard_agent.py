import sys
import numpy as np
import h5py
import click
import json
import pyglet
from ai2thor.controller import Controller
from PIL import Image
from topview import *

ALL_POSSIBLE_ACTIONS = [
	'MoveAhead',
	'MoveBack',
	'RotateRight',
	'RotateLeft',
	# 'Stop'   
]

class SimpleImageViewer(object):

  def __init__(self, display=None):
    self.window = None
    self.isopen = False
    self.display = display

  def imshow(self, arr):
    if self.window is None:
      height, width, channels = arr.shape
      self.window = pyglet.window.Window(width=width, height=height, display=self.display, caption="THOR Browser")
      self.width = width
      self.height = height
      self.isopen = True

    assert arr.shape == (self.height, self.width, 3), "You passed in an image with the wrong number shape"
    image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes(), pitch=self.width * -3)
    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()
    image.blit(0,0)
    self.window.flip()


  def close(self):
    if self.isopen:
      self.window.close()
      self.isopen = False

  def __del__(self):
    self.close()

class TopViewImageViewer(object):

  def __init__(self, display=None):
    self.window = None
    self.isopen = False
    self.display = display

  def imshow(self, arr):
    if self.window is None:
      height, width, channels = arr.shape
      self.window = pyglet.window.Window(width=width, height=height, display=self.display, caption="THOR Browser")
      self.width = width
      self.height = height
      self.isopen = True

    assert arr.shape == (self.height, self.width, 3), "You passed in an image with the wrong number shape"
    image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes(), pitch=self.width * -3)
    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()
    image.blit(0,0)
    self.window.flip()


  def close(self):
    if self.isopen:
      self.window.close()
      self.isopen = False

  def __del__(self):
    self.close()

def key_press(key, mod):
	global human_agent_action, human_wants_restart, stop_requested
	if key == ord('R') or key == ord('r'): # r/R
		human_wants_restart = True
	if key == ord('Q') or key == ord('q'): # q/Q
		stop_requested = True

	if key == 0xFF52: # move ahead
		human_agent_action = 0
	if key == 0xFF54: # move back
		human_agent_action = 1
	if key == 0xFF53: # turn right
		human_agent_action = 2
	if key == 0xFF51: # turn left
		human_agent_action = 3

	if key == ord('z'): # look down
		human_agent_action = 4
	if key == ord('x'): # look up
		human_agent_action = 5

if __name__ == '__main__':
	
	# run()

	angle = 45.0

	human_agent_action = None
	human_wants_restart = False
	stop_requested = False
	next_position = None
	visible = None

	f = h5py.File('dumped/FloorPlan317.hdf5', "r")
	observations = f['observations']
	graph = f['graph']
	visible_objects = f['visible_objects']
	dump_features = f['dump_features']
	states = f['locations'][()]

	config = json.load(open('config.json'))
	categories = list(config['new_objects'].keys())

	k = np.random.randint(0, observations.shape[0])
	while states[k][2] % angle != 0.0:
		k = np.random.randint(0, observations.shape[0])
	current_position = k

	viewer = SimpleImageViewer()
	viewer.imshow(observations[current_position].astype(np.uint8))
	viewer.window.on_key_press = key_press

	# -----------
	controller = Controller()
	controller.start()
	controller.reset("FloorPlan317")

	drawer = TrajectoryDrawer(controller)
	new_frame = drawer.draw({'x':states[k][0],'z':states[k][1]})

	topviewer = TopViewImageViewer()
	topviewer.imshow(new_frame.astype(np.uint8))
	topviewer.window.on_key_press = key_press
	#--------------

	print("Use arrow keys to move the agent.")
	print("Press R to reset agent\'s location.")
	print("Press Q to quit.")

	while True:
		# waiting for keyboard input
		if human_agent_action is not None:
			# move actions
			if human_agent_action == 2 or human_agent_action == 3:
				next_position = current_position
				for _ in range(int(angle/ 22.5)):
					next_position = graph[next_position][human_agent_action]
			else:
				next_position = graph[current_position][human_agent_action]
				
			current_position = next_position if next_position != -1 else current_position
			distances = [(categories[i], dump_features[current_position][i]) for i in list(np.where(dump_features[current_position][:-4] > 0)[0])]
			print(distances, dump_features[current_position][-4:])
			visible = visible_objects[current_position].split(',')
			human_agent_action = None

		# waiting for reset command
		if human_wants_restart:
			# reset agent to random location
			k = np.random.randint(0, observations.shape[0])
			while states[k][2] % angle != 0.0:
				k = np.random.randint(0, observations.shape[0])
			current_position = k

			human_wants_restart = False

		# check collision
		if next_position == -1:
			print('Collision occurs.')

		# check quit command
		if stop_requested: break

		new_frame = drawer.draw({'x': states[current_position][0], 'z': states[current_position][1]})
		topviewer.imshow(new_frame.astype(np.uint8))
		viewer.imshow(observations[current_position].astype(np.uint8))


		if visible is not None and len(list(visible)) > 0:
			print("Visible: {}".format(visible))
			visible = None

	print("Goodbye.")
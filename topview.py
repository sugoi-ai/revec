import matplotlib
from ai2thor.controller import Controller

import numpy as np
import matplotlib.pyplot as plt
import cv2


class TrajectoryDrawer(object):
    def __init__(self, controller, cmap='viridis'):

        event = controller.step({"action": "ToggleMapView"})

        cam_position = event.metadata["cameraPosition"]
        orth_size = event.metadata["cameraOrthSize"]

        self.topdown_map = event.frame
        self.topdown_map_semantic = event.class_segmentation_frame
        self.map_height, self.map_width = event.frame.shape[:2]
        self.lower_left = self._convert_position(cam_position) - orth_size
        self.span = 2 * orth_size
        self.base_heat = 0
        self.heatmap = np.zeros((self.map_height, self.map_width)) + self.base_heat
        self.radious = self.map_width / 8 * 0.4
        self.cmap = plt.get_cmap(cmap)

        controller.step({"action": "ToggleMapView"})

    def _translate_position(self, position):
        position = self._convert_position(position)

        camera_position = (position - self.lower_left) / self.span

        return np.array([round(self.map_width * camera_position[0]),
                         round(self.map_height * (1.0 - camera_position[1]))],
                        dtype=int)

    def _convert_position(self, position):
        return np.array([position["x"], position["z"]])

    def _update_heatmap(self, agent_position):
        x, y = self._translate_position(agent_position)

        span_y, span_x = np.ogrid[-y : self.map_height - y,
                                  -x : self.map_width - x]

        mask = span_x**2 + span_y**2 <= self.radious**2

        max_heat = np.max(self.heatmap)
        if max_heat == 0:
            self.heatmap[mask] = 1
        else:
            self.heatmap[mask] = max_heat * 1.1

        self.heatmap /= np.max(self.heatmap)

    def draw(self, agent_position):
        self._update_heatmap(agent_position)

        overlay = self.cmap(self.heatmap)
        overlay = (overlay[..., :3] * 255).astype('int')

        normal_frame = self.topdown_map.copy()
        x, y = self._translate_position(agent_position)
        new_frame = cv2.circle(normal_frame, (x,y),10,(153,0,153),-1)

        return new_frame
if __name__ == "__main__":
    controller = Controller()
    controller.start()
    controller.reset("FloorPlan2")

    drawer = TrajectoryDrawer(controller)
    new_frame = drawer.draw(controller.last_event.metadata["agent"]["position"])
    print(controller.last_event.metadata["agent"]["position"])
    plt.imshow(new_frame)
plt.show()
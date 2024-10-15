import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import json

class PredictorVectorfield():

    def __init__(self, XY_forward:tuple = None, XY_backward:tuple = None, voxel=(1000000,2,0.347,0.347), path:str = None):

        # print(XY_forward[0][0,:])
        # print(XY_forward[1][0,:])

        self.loaded = False
        if not XY_forward is None or not XY_forward is None :

            self.voxel = np.array(voxel)

            self.forward = not XY_forward is None
            self.backward = not XY_backward is None

            if self.forward:
                self.X_forward = XY_forward[0]*self.voxel
                self.Y_forward = XY_forward[1]*self.voxel

            if self.backward:
                self.X_backward = XY_backward[0]*self.voxel
                self.Y_backward = XY_backward[1]*self.voxel

        elif not path is None:

            with open("{}/settings.json".format(path), 'r') as settings_file:
                settings = json.load(settings_file)

            self.forward = settings["forward"]
            self.backward = settings["backward"]
            self.voxel = np.array(settings["voxel"])

            if self.forward:
                self.X_forward = np.load("{}/X_forward.npy".format(path))
                self.Y_forward = np.load("{}/Y_forward.npy".format(path))
            
            if self.backward:
                self.X_backward = np.load("{}/X_backward.npy".format(path))
                self.Y_backward = np.load("{}/Y_backward.npy".format(path))

        else:

            raise Exception("or XY_forward/XY_backward or path should be specified")

        self.t_max = int(np.max(self.Y_forward[:,0])/self.voxel[0])

        if not self.forward:
            self.X_forward = self.Y_backward
            self.Y_forward = self.X_backward

        if not self.backward:
            self.X_backward = self.Y_forward
            self.Y_backward = self.X_forward

        self.model_forward = KNeighborsRegressor(n_neighbors=8)
        self.model_forward.fit(self.X_forward, self.Y_forward)
        self.model_backward = KNeighborsRegressor(n_neighbors=8)
        self.model_backward.fit(self.X_backward, self.Y_backward)

    def predict(self, point, forward, n_steps=-1):

        if n_steps == -1:
            n_steps = np.inf

        t = int(point[0])

        if forward:
            model = self.model_forward
            n_iterations = max(min(self.t_max - t, n_steps),0)
        else:
            model = self.model_backward
            n_iterations = max(min(t, n_steps),0)

        predictions = np.zeros([n_iterations,4])

        point_new = point.copy().reshape(1,-1)*self.voxel
        for i in range(n_iterations):
            point_new = model.predict(point_new)
            predictions[i,:] = point_new[0,:]/self.voxel

        return predictions

    def save(self, path):

        with open("{}/settings.json".format(path), 'w') as settings_file:
            json.dump(
                {
                    "voxel":[float(i) for i in self.voxel],
                    "forward":self.forward,
                    "backward":self.backward,
                    "t_max":float(self.t_max)
                }, settings_file, indent=4)
        
        if self.forward:
            np.save("{}/X_forward.npy".format(path), self.X_forward)
            np.save("{}/Y_forward.npy".format(path), self.Y_forward)

        if self.backward:
            np.save("{}/X_backward.npy".format(path), self.X_backward)
            np.save("{}/Y_backward.npy".format(path), self.Y_backward)
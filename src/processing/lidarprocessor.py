import copy
import json
import numpy as np
import time
import os
from .pcapframeparser import PcapFrameParser
from .pcdframeparser import PcdFrameParser
from .framestream import FrameStream
from .planeprocessor import PlaneProcessor
from .cloudclipper import CloudClipper
from .backgroundextractor import BackgroundExtractor
from .backgroundsubtractor import BackgroundSubtractor, ElementComparisonSubtractor
from .dataentities import Frame
from .clusterer import Clusterer
from .tracker import Tracker

class LidarProcessor():
    def __init__(self):
        self.filename = None
        self._originalFrames = []
        self._timestamps = []
        self._preprocessedArrays = []
        self._preprocessedArraysTemp = []

        self.frameGenerator = None

        self.plane_processor = None

        self.clipper = None
        
        self.bg_subtractor = None
        self.bg_extractor = None
        self.bg_filename = None
        self.originalBgFrame = None
        self.preprocessedBgArray = None
        self.numScanLines = None
        self.range_thrld_matrix = None

        self.clusterer = None
        self.frameClusters = []

        self.tracker = None
    #
    # LOAD/SAVE config
    #
    def init_from_config(self, configpath):
        with open(configpath, "r") as read_file:
            config = json.load(read_file)

        # need validator for the config file to check if all necessary fields
        # are there and spelled correctly

        # call processors one by one
        # plane processor
        if "ground" in config.keys():
            self.createPlaneProcessor(method=config["ground"]["method"], 
                **config["ground"]["params"])
        else:
            self.destroyPlaneProcessor()

        # clipper
        if "clipper" in config.keys():
            self.createClipper(method=config["clipper"]["method"], 
                **config["clipper"]["params"])
        else:
            self.destroyClipper()

        # background cloud and subtractor
        if "background" in config.keys():
            isLoaded = True
            if "path" in config["background"]:
                # try to get background cloud by loading file
                bg_path = config["background"]["path"]
                if os.path.exists(bg_path):
                    self.loadBackground(bg_path, method=config["background"]["subtractor"]["method"])
                else:
                    isLoaded = False

            if "extractor" in config["background"] and not isLoaded:
                # if no path or invalid path, try to extract cloud
                self.extractBackground(
                    method=config["background"]["extractor"]["method"],
                    numScanLines=config["background"]["numScanLines"] if "numScanLines" in config["background"] else 32,
                    **config["background"]["extractor"]["params"])

            # finally create bg subtractor
            if "subtractor" in config["background"]:
                self.createBgSubtractor(
                    method=config["background"]["subtractor"]["method"],
                    **config["background"]["subtractor"]["params"])
        else:
            self.destroyBgExtractor()
            self.destroyBgSubtractor()
        
        #clustering
        if "clustering" in config.keys():
            method = config["clustering"]["method"]
            params = config["clustering"]["params"]
            self.createClusterer(method, **params)
        else:
            self.destroyClusterer()

        if "tracking" in config.keys():
            method = config["tracking"]["method"]
            params = config["tracking"]["params"]
            self.createTracker(method, **params)
        else:
            self.destroyTracker()

    def save_config(self, configpath):
        with open(configpath, "w") as write_file:

            config = {}
            # plane processor pre-processor part
            if self.plane_processor is not None:
                settings = self.plane_processor.get_config()
                config["ground"] = settings

            # clipper pre-processor part
            if self.clipper is not None:
                settings = self.clipper.get_config()
                config["clipper"] = settings

            # background pre-processor part
            config["background"] = {}
            if self.bg_filename is not None:
                config["background"]["path"] = self.bg_filename
            else:
                config["background"]["path"] = ""

            config["background"]["numScanLines"] = self.numScanLines

            if self.bg_extractor is not None:
                settings = self.bg_extractor.get_config()
                config["background"]["extractor"] = settings

            if self.bg_subtractor is not None:
                settings = self.bg_subtractor.get_config()
                config["background"]["subtractor"] = settings

            if self.clusterer:
                settings = self.clusterer.get_config()
                config["clustering"] = settings

            if self.tracker:
                settings = self.tracker.get_config()
                config["tracking"] = settings

            json.dump(config, write_file, indent=4)
            print("Config saved to:\n{0}".format(configpath))

    def resetProcessors(self):
        self.destroyPlaneProcessor()
        self.destroyClipper()
        self.destroyBgExtractor()
        self.destroyBgSubtractor()
        self.destroyClusterer()
        self.destroyTracker()
    #
    # I/O
    #
    def setFilename(self, filename):
        self.filename = filename

    def restartBuffering(self):
        if self.filename is None:
            return

        if isinstance(self.filename, list) and os.path.splitext(self.filename[0])[1] == '.pcd':
            parser = PcdFrameParser(self.filename)
        else:
            parser = PcapFrameParser(self.filename[0])
        self.frameGenerator = parser.generator()

    def loadNFrames(self, N):
        self._timestamps = []
        self._originalFrames = []
        self._preprocessedArrays = []
        for i in range(N):
            (ts, f) = self.readNextFrame()
            self._timestamps.append(ts)
            self._originalFrames.append(f)

            # preprocessed frames are the cartesian xyz arrays
            pts = self.arrayFromFrame(f)
            self._preprocessedArrays.append(pts)

    def readNextFrame(self):
        try:
            out = next(self.frameGenerator)
        except StopIteration:
            out = (None, None)
        return out

    def peek_size(self):
        if isinstance(self.filename, list) and os.path.splitext(self.filename[0])[1] == '.pcd':
            return len(self.filename)
        else:
            parser = PcapFrameParser(self.filename[0])
            return parser.peek_size()

    # test stuff
    def resetFrameData(self):
        self._timestamps = []
        self._originalFrames = []
        self._preprocessedArrays = []

    def loadFrame(self):
        (ts, f) = self.readNextFrame()
        self._timestamps.append(ts)
        self._originalFrames.append(f)

        # preprocessed frames are the cartesian xyz arrays
        pts = self.arrayFromFrame(f)
        self._preprocessedArrays.append(pts)

    def getTimestamp(self, frameID):
        return self._timestamps[frameID]

    def getArray(self,frameID):
        return self._preprocessedArrays[frameID]

    def arrayFromFrame(self, frame):
        # TODO: Use always getCartesianAccordingToMatlab if does not conflict with baseline
        if isinstance(self.filename, list) and os.path.splitext(self.filename[0])[1] == '.pcd':
            x,y,z = frame.getCartesianAccordingToMatlab()
        else:
            x,y,z = frame.getCartesian()
        pts = np.vstack((x,y,z)).astype(np.float32).T
        pts = self.removeZeros(pts)
        return pts

    # 
    # PREPROCESSING
    #
    def updateBackground(self):
        # ensure the subtractor is initiated with the correct
        # background point cloud
        if self.originalBgFrame is not None:
            if self.bg_subtractor is not None:
                if isinstance(self.bg_subtractor, ElementComparisonSubtractor):
                    #  TODO: until now work-around: frame at night as static
                    # background because sph2cart transformation with 
                    # range_thrld_matrix as distance leads to skewed point cloud
                    pts = self.arrayFromFrame(self.originalBgFrame)
                    self.preprocessedBgArray = pts
                    self.bg_subtractor.set_background(self.preprocessedBgArray)
                    self.bg_subtractor.set_range_thrld_matrix(self.range_thrld_matrix)
                else:
                    pts = self.arrayFromFrame(self.originalBgFrame)
                    self.preprocessedBgArray = self.preprocessBg(pts)
                    self.bg_subtractor.set_background(self.preprocessedBgArray)
            else:
                pts = self.arrayFromFrame(self.originalBgFrame)
                self.preprocessedBgArray = self.preprocessBg(pts)

    def updatePreprocessedGen(self):
        self.updateBackground()
        # update preprocessed points
        self._preprocessedArrays = []
        for (i, frame) in enumerate(self._originalFrames):
            pts = self.arrayFromFrame(frame)
            pts = self.preprocessArray(pts)
            self._preprocessedArrays.append(pts)

            # yield completion status
            yield (i+1)*100/(len(self._originalFrames))

    def updatePreprocessed(self):
        self.updateBackground()

        # update preprocessed points
        self._preprocessedArrays = []
        for (i, frame) in enumerate(self._originalFrames):
            pts = self.arrayFromFrame(frame)
            pts = self.preprocessArray(pts)
            self._preprocessedArrays.append(pts)

    def preprocessArray(self, arr):
        if isinstance(self.filename, list) and os.path.splitext(self.filename[0])[1] == '.pcd':
            import open3d as o3d
            pc = o3d.geometry.PointCloud()
            # extract front sensor area
            arr = arr[arr[:, 0] > 0]

            # remove sensor station artifacts
            pc.points = o3d.utility.Vector3dVector(arr)
            min_bound = np.array([-5, -17.5, -10])
            max_bound = np.array([7, 19, 5])
            sensor_station = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            pc = pc.select_by_index(sensor_station.get_point_indices_within_bounding_box(pc.points), invert=True)
            arr = np.asarray(pc.points)

        # apply plane processor
        if self.plane_processor is not None:
            arr = self.plane_processor.apply(arr)
            
        # apply clipper
        if self.clipper is not None:
            arr = self.clipper.clip(arr)

        # apply bg subtractor
        if self.bg_subtractor is not None:
            arr = self.bg_subtractor.subtract(arr)

        # TODO: Make this available in config file
        if isinstance(self.filename, list) and os.path.splitext(self.filename[0])[1] == '.pcd':
            import open3d as o3d
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(arr)
            _, indices = pc.remove_radius_outlier(nb_points=30, radius=2.0)
            indices = np.array(indices)
            pc_filtered = pc.select_by_index(indices)
            arr = np.asarray(pc_filtered.points)

        return arr

    def preprocessBg(self, arr):
        # apply plane processor
        if self.plane_processor is not None:
            arr = self.plane_processor.apply(arr)
            
        # apply clipper
        if self.clipper is not None:
            arr = self.clipper.clip(arr)

        return arr

    def removeZeros(self, arr):
        return arr[np.any(arr, axis=1)]
    #
    # PLANE ESTIMATION
    #
    def destroyPlaneProcessor(self):
        self.plane_processor = None

    def createPlaneProcessor(self, method, points=None, **kwargs):
            if points is not None:
                self.plane_processor = PlaneProcessor.factory("3_points_plane")
                self.plane_processor.get_plane_from_3_points(points)
            elif "normal" in kwargs and "intercept" in kwargs:
                self.plane_processor = PlaneProcessor.factory(method,
                    **kwargs)
            else:
                ValueError("")

    def getPlaneCoeff(self):
        return self.plane_processor.get_plane_coeff()

    #
    # CLOUD CLIPPER
    #
    def createClipper(self, method, **kwargs):
        self.clipper = CloudClipper.factory(method, **kwargs)

    def destroyClipper(self):
        self.clipper = None

    #
    # BG SUBTRACTOR / EXTRACTOR
    #
    def saveBackground(self, filename):
        if self.originalBgFrame is not None:
            self.originalBgFrame.save_csv(filename)
            self.bg_filename = filename

    def loadBackground(self, filename, method="kd_tree"):
        self.bg_extractor = None
        self.originalBgFrame = Frame()
        self.originalBgFrame.load_csv(filename)
        self.bg_filename = filename
        pts = self.arrayFromFrame(self.originalBgFrame)
        self.preprocessedBgArray = self.preprocessArray(pts)
        if method == "element_comparison":
            range_thrld_matrix_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data/range_thrld_matrix.npy")
            try:
                self.range_thrld_matrix = np.load(range_thrld_matrix_path)
            except FileNotFoundError:
                print(f'{range_thrld_matrix_path} does not exist.')
            

    def extractBackground(self, method, numScanLines, **kwargs):
        self.numScanLines = numScanLines
        self.bg_extractor = BackgroundExtractor(self.numScanLines, **kwargs)
        self.bg_extractor.extract(self._originalFrames)
        self.originalBgFrame = self.bg_extractor.get_background()
        #
        pts = self.arrayFromFrame(self.originalBgFrame)
        self.preprocessedBgArray = self.preprocessArray(pts)

    def destroyBgExtractor(self):
        self.bg_extractor = None
        self.originalBgFrame = None
        self.preprocessedBgArray = None
        self.numScanLines = None

    def createBgSubtractor(self, method, **kwargs):
        if self.originalBgFrame is not None:
            self.bg_subtractor = BackgroundSubtractor.factory(method, **kwargs)

    def destroyBgSubtractor(self):
        self.bg_filename = None
        self.bg_subtractor = None

    #
    # CLUSTERER
    #
    def getClusters(self, frameID):
        if self.frameClusters:
            return self.frameClusters[frameID]
        else:
            return []

    def createClusterer(self, method, **kwargs):
        self.clusterer = Clusterer.factory(method, **kwargs)
        self.frameClusters = []

    def extractClustersGen(self):
        self.frameClusters = []
        for i, arr in enumerate(self._preprocessedArrays):
            clusters = self.clusterer.cluster(arr)
            self.frameClusters.append(clusters)
            
            # yield completion status
            yield (i+1)*100/(len(self._originalFrames))

    def destroyClusterer(self):
        self.frameClusters = []
        self.clusterer = None

    #
    # TRACKING
    #
    def createTracker(self, method, **kwargs):
        self.tracker = Tracker.factory(method, **kwargs)

    def trackClustersGen(self):
        self.tracker.restart()
        for i, clusters in enumerate(self.frameClusters):
            # TODO: Guarantee that centroid is calculated on creation
            centroids = [c.centroid for c in clusters]
            self.tracker.update(centroids)

            mapping = self.tracker.getInputMapping()
            for j, c in enumerate(clusters):
                c.id = mapping[j]

            # yield completion status
            yield (i+1)*100/(len(self._originalFrames))

    def destroyTracker(self):
        self.tracker = None
        for clusters in self.frameClusters:
            for c in clusters:
                c.id = None

    #
    # Entire pipeline
    #
    def get_status(self):
        status = {
            "plane_process": self.plane_processor is not None,
            "clipping": self.clipper is not None,
            "background_extraction": self.bg_extractor is not None,
            "background_subtraction": self.bg_subtractor is not None,
            "clustering": self.clusterer is not None,
            "tracking": self.tracker is not None,
        }
        return status


    def updateProcessingGen(self):
        self.updateBackground()
        # update preprocessed points
        self._preprocessedArrays = []
        self.frameClusters = []

        if self.tracker:
            self.tracker.restart()

        for (i, frame) in enumerate(self._originalFrames):
            pts = self.arrayFromFrame(frame)
            pts = self.preprocessArray(pts)
            self._preprocessedArrays.append(pts)

                # apply clustering
            if self.clusterer:
                clusters = self.clusterer.cluster(pts)
                self.frameClusters.append(clusters)

                # apply tracking
            if self.tracker:
                centroids = [c.centroid for c in clusters]
                self.tracker.update(centroids)
                mapping = self.tracker.getInputMapping()
                for j, c in enumerate(clusters):
                    c.id = mapping[j]

            # yield completion status
            yield (i+1)*100/(len(self._originalFrames))

    def processingGen(self, start_frame, end_frame):
        if self.clusterer is None or self.tracker is None:
            return

        self.tracker.restart()#? or just reinit in controller
        for i in range(0, end_frame + 1):
            if i < start_frame:
                continue

            (ts, frame) = self.readNextFrame()

            # apply plane process, clipping, bg subtraction
            pts = self.arrayFromFrame(frame)
            pts = self.preprocessArray(pts)

            # apply clustering
            clusters = self.clusterer.cluster(pts)

            # apply tracking
            centroids = [c.centroid for c in clusters]
            self.tracker.update(centroids)
            mapping = self.tracker.getInputMapping()
            for j, c in enumerate(clusters):
                c.id = mapping[j]

            # output frame number, time, tracked clusters an dprogress
            p = (i-start_frame+1)*100/(end_frame - start_frame)
            yield (i, ts, clusters, p)


# def cart2sph(x: np.ndarray, y: np.ndarray,
#              z: np.ndarray):
#     """ Convert cartesian to spherical coordinates """

#     # Convert -0.0 in 0.0 because np.arctan2() distinguishes 0.0 and -0.0
#     x = x + 0.0
#     y = y + 0.0
#     z = z + 0.0

#     hypotxy = np.hypot(x, y)
#     r = np.hypot(hypotxy, z)
#     elevation = np.arctan2(z, hypotxy)
#     azimuth = np.arctan2(y, x)
#     return azimuth, elevation, r


# if __name__ == '__main__':
#     import os
#     from pypcd import pypcd
#     from dataentities import Frame
#     import open3d as o3d

#     settings = {
#         "path": "",
#         "extractor":{
#             "method":"range_image",
#             "params":{
#                 "percentile": 0.7,
#                 "non_zero": 0.8,
#                 "n_frames": 90
#             }
#         },
#         "subtractor":{
#             "method": "kd_tree", # "kd_tree", "octree"
#             "params": {
#                 "search_radius": 0.05 # "search_radius": 0.2, "resolution": 0.01
#                 }
#             }
#     }

#     dataset_path = '/home/marcel/Repositorys/a9_dataset_r01_s04/_points'

#     frame_list = os.listdir(dataset_path)
#     frame_list.sort()
#     originalFrames = []

#     for frame_num, frame in enumerate(frame_list):
#         print(f'Processing frame {frame_num} ...')
#         input_file = os.path.join(dataset_path, frame)
#         pc = pypcd.PointCloud.from_path(input_file)
#         if 'ring' not in pc.fields:
#             print(f'\nDoes not contain ring information: {input_file}')
#             continue
#         azimuth, elevation, range_from_sensor = cart2sph(pc.pc_data['x'], pc.pc_data['y'], pc.pc_data['z'])
#         frame = np.hstack((pc.pc_data['ring'][:, None],
#                            elevation[:, None],
#                            azimuth[:, None],
#                            range_from_sensor[:, None],
#                            pc.pc_data['intensity'][:, None]))
#         originalFrames.append(frame)
#     originalFrames = np.array(originalFrames)
#     np.save("/home/marcel/Repositorys/pylidartracker/src/processing/a9_dataset_r01_s04.npy", originalFrames)

#     originalFrames = []
#     originalFrames_numpy = np.load("/home/marcel/Repositorys/pylidartracker/src/processing/a9_dataset_r01_s04.npy")
#     for i in range(originalFrames_numpy.shape[0]):
#         frame = Frame()
#         frame.id = originalFrames_numpy[i, :, 0]
#         frame.elevation = originalFrames_numpy[i, :, 1]
#         frame.azimuth = originalFrames_numpy[i, :, 2]
#         frame.distance = originalFrames_numpy[i, :, 3]
#         frame.intensity = originalFrames_numpy[i, :, 4]
#         originalFrames.append(frame)
#     model = LidarProcessor()
#     model._originalFrames = originalFrames

#     model.extractBackground(method=settings["extractor"]["method"], **settings["extractor"]["params"])
#     model.createBgSubtractor(method=settings["subtractor"]["method"], **settings["subtractor"]["params"])

#     model.saveBackground("/home/marcel/Repositorys/pylidartracker/a9_background.txt")

#     model.updatePreprocessed()

#     # Visualization
#     pc = o3d.geometry.PointCloud()
#     for i in range(len(model._originalFrames)):
#         print(f'Saving frame {i} ...')
#         points = model._preprocessedArrays[i]
#         pc.points = o3d.utility.Vector3dVector(points)
#         pc.paint_uniform_color([1, 0, 0])
#         o3d.io.write_point_cloud("/home/marcel/Repositorys/pylidartracker/a9_preprocessed/frame_" + str(i) + ".pcd", pc, write_ascii=True)

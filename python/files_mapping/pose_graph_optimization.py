import g2o
import numpy as np
import matplotlib.pyplot as plt

# from hd_map.common.meshlab import MeshlabInf


class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        self.edge_vertices = set()

        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

        # See https://github.com/RainerKuemmerle/g2o/issues/34
        self.se3_offset_id = 0
        se3_offset = g2o.ParameterSE3Offset()
        se3_offset.set_id(self.se3_offset_id)
        super().add_parameter(se3_offset)

    def optimize(self, max_iterations=20, verbose=False):
        super().initialize_optimization()
        super().set_verbose(verbose)
        super().optimize(max_iterations)

    def add_vertex(self, vertex_id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(vertex_id)
        v_se3.set_estimate(g2o.Isometry3d(pose))
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_vertex_point(self, vertex_id, point, fixed=False):
        v_point = g2o.VertexPointXYZ()
        v_point.set_id(vertex_id)
        v_point.set_estimate(point)
        v_point.set_fixed(fixed)
        super().add_vertex(v_point)

    def add_edge(self, vertices, measurement, information=np.eye(6), robust_kernel=None):
        self.edge_vertices.add(vertices)

        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(g2o.Isometry3d(measurement))  # relative pose
        edge.set_information(information)
        # robust_kernel = g2o.RobustKernelHuber()
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def add_edge_pose_point(self, vertex_pose, vertex_point, measurement, information=np.eye(3), robust_kernel=None):
        edge = g2o.EdgeSE3PointXYZ()
        edge.set_vertex(0, self.vertex(vertex_pose))
        edge.set_vertex(1, self.vertex(vertex_point))
        edge.set_measurement(measurement)
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        edge.set_parameter_id(0, self.se3_offset_id)
        super().add_edge(edge)

    def get_pose(self, vertex_id):
        return self.vertex(vertex_id).estimate().matrix()

    def does_edge_exists(self, vertex_id_a, vertex_id_b):
        return (vertex_id_a, vertex_id_b) in self.edge_vertices or (vertex_id_b, vertex_id_a) in self.edge_vertices

    def is_vertex_in_any_edge(self, vertex_id):
        vertices = set()
        for edge in self.edge_vertices:
            vertices.add(edge[0])
            vertices.add(edge[1])
        return vertex_id in vertices

    def does_vertex_have_only_global_edges(self, vertex_id):
        assert self.is_vertex_in_any_edge(vertex_id)
        for edge in self.edge_vertices:
            if vertex_id not in edge:
                continue
            if np.abs(edge[0] - edge[1]) == 1:
                return False
        return True

    def visualize_in_meshlab(self, filename, meshlab=None):
        points = {}
        for vertex_id, vertex in self.vertices().items():
            if isinstance(vertex, g2o.VertexSE3):
                points[vertex_id] = vertex.estimate().matrix()[:3, 3]

        if meshlab is None:
            meshlab = MeshlabInf()
        for point in points.values():
            meshlab.add_points(point)
        for edge in self.edge_vertices:
            meshlab.add_line(points[edge[0]], points[edge[1]])
        meshlab.write(filename)


    def visualize_in_plt(self, threeDim = False):
        points = {}
        for vertex_id, vertex in self.vertices().items():
            if isinstance(vertex, g2o.VertexSE3):
                points[vertex_id] = vertex.estimate().matrix()[:3, 3]
        x = []
        y = []
        z = []
        for i in range(len(points)):
            x.append(points[i][0])
            y.append(points[i][1])
            z.append(points[i][2])

        fig = plt.figure()
        if threeDim:
            ax = fig.gca(projection='3d')
            ax.plot(x, y, z)
            # Plot scatterplot data (20 2D points per colour) on the x and z axes.
            c_list = np.linspace(0, 1, len(points))
            # By using zdir='y', the y value of these points is fixed to the zs value 0
            # and the (x,y) points are plotted on the x and z axes.
            ax.scatter(x, y, z, c=c_list, label='points in (x,z)')

            # # Make legend, set axes limits and labels
            # ax.legend()
            # ax.set_xlim(0, 1)
            # ax.set_ylim(0, 1)
            # ax.set_zlim(0, 1)
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            #
            # # Customize the view angle so it's easier to see that the scatter points lie
            # # on the plane y=0
            # ax.view_init(elev=20., azim=-35)
        else:
            plt.plot(x, y)
            c_list = np.linspace(0, 1, len(points))
            plt.scatter(x, y, c=c_list, label='points in (x,z)')
        plt.show()